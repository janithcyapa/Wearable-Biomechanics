"""PySide6 pop-out interactive frame-by-frame Video Blur Fixer.

Can be launched from Jupyter (`launch_blur_editor`) or CLI:
    python -m src.export.video_blur_ui /path/to/videos_or_video.mp4
"""

from __future__ import annotations

import copy
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.export.video_blur_core import (
    BACKUP_DIR_NAME,
    BlurProject,
    ExportResult,
    apply_blur_to_frame,
    export_blurred_video,
    interpolate_track_at_frame,
    load_qa_state,
    load_recipe,
    probe_video,
    render_mask_for_frame,
    save_recipe,
    set_video_qa_status,
)


def launch_blur_editor(
    target_path: str | Path,
    wait: bool = True,
) -> subprocess.CompletedProcess | subprocess.Popen:
    """Launch the PySide6 pop-out brush editor in an isolated subprocess.

    Running in a subprocess avoids Qt event-loop conflicts inside Jupyter/VS Code.
    """
    path = Path(target_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    cmd = [sys.executable, "-m", "src.export.video_blur_ui", str(path)]
    if wait:
        return subprocess.run(cmd, check=False)
    return subprocess.Popen(cmd)


# ---------------------------------------------------------------------------
# PySide6 GUI Implementation (imported lazily so headless callers work)
# ---------------------------------------------------------------------------


def _build_gui_classes():
    from PySide6.QtCore import QPoint, QRect, Qt, QThread, QTimer, Signal
    from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSlider,
        QSpinBox,
        QStatusBar,
        QVBoxLayout,
        QWidget,
    )

    class ExportWorker(QThread):
        progress = Signal(int, int)
        finished_ok = Signal(object)
        failed = Signal(str)

        def __init__(self, video_path: Path, project: BlurProject):
            super().__init__()
            self.video_path = video_path
            self.project = project

        def run(self) -> None:
            try:
                res = export_blurred_video(
                    self.video_path,
                    self.project,
                    progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
                )
                self.finished_ok.emit(res)
            except Exception as exc:  # pragma: no cover - surfaced in dialog
                self.failed.emit(str(exc))

    class VideoCanvas(QWidget):
        """Interactive video frame canvas with scaled coordinate mapping & brush overlay."""

        stroke_started = Signal(float, float)
        stroke_moved = Signal(float, float)
        stroke_finished = Signal()

        def __init__(self, parent: QWidget | None = None):
            super().__init__(parent)
            self.setMouseTracking(True)
            self.setMinimumSize(640, 360)
            self.setStyleSheet("background-color: #111318;")

            self._pixmap: QPixmap | None = None
            self._vid_w: int = 1
            self._vid_h: int = 1
            self._draw_rect = QRect()
            self._cursor_pos: QPoint | None = None
            self._brush_radius_px: int = 40
            self._is_eraser: bool = False
            self._drawing: bool = False
            self._active_kf_circles: list[tuple[float, float, float, bool, str]] = []

        def set_frame_rgb(
            self,
            rgb_img: np.ndarray,
            kf_circles: list[tuple[float, float, float, bool, str]] | None = None,
        ) -> None:
            h, w, _ = rgb_img.shape
            self._vid_w = w
            self._vid_h = h
            qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._pixmap = QPixmap.fromImage(qimg)
            self._active_kf_circles = kf_circles or []
            self.update()

        def set_brush_state(self, radius_px: int, is_eraser: bool) -> None:
            self._brush_radius_px = max(1, int(radius_px))
            self._is_eraser = is_eraser
            self.update()

        def _compute_draw_rect(self) -> QRect:
            if self._pixmap is None or self._vid_w <= 0 or self._vid_h <= 0:
                return QRect(0, 0, self.width(), self.height())
            ww, wh = self.width(), self.height()
            scale = min(ww / self._vid_w, wh / self._vid_h)
            dw = int(round(self._vid_w * scale))
            dh = int(round(self._vid_h * scale))
            ox = (ww - dw) // 2
            oy = (wh - dh) // 2
            return QRect(ox, oy, dw, dh)

        def _widget_to_video(self, pos: QPoint) -> tuple[float, float] | None:
            r = self._draw_rect
            if r.width() <= 0 or r.height() <= 0:
                return None
            vx = (pos.x() - r.x()) * (self._vid_w / float(r.width()))
            vy = (pos.y() - r.y()) * (self._vid_h / float(r.height()))
            vx = max(0.0, min(float(self._vid_w - 1), vx))
            vy = max(0.0, min(float(self._vid_h - 1), vy))
            return vx, vy

        def paintEvent(self, event) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#111318"))

            if self._pixmap is not None:
                self._draw_rect = self._compute_draw_rect()
                painter.drawPixmap(self._draw_rect, self._pixmap)

                scale = self._draw_rect.width() / float(self._vid_w)
                # Draw keyframe track guides
                for cx, cy, rad, is_exact_kf, label in self._active_kf_circles:
                    wx = self._draw_rect.x() + cx * scale
                    wy = self._draw_rect.y() + cy * scale
                    wr = rad * scale
                    pen = QPen(QColor("#00e5ff") if is_exact_kf else QColor("#ffd54f"), 2)
                    if not is_exact_kf:
                        pen.setStyle(Qt.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(QPoint(int(wx), int(wy)), int(wr), int(wr))
                    painter.drawText(int(wx - wr), int(wy - wr - 4), label)

                # Draw brush cursor
                if self._cursor_pos is not None and self._draw_rect.contains(self._cursor_pos):
                    cr = max(2, int(round(self._brush_radius_px * scale)))
                    color = QColor("#ff5252") if self._is_eraser else QColor("#69f0ae")
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(self._cursor_pos, cr, cr)

            painter.end()

        def mousePressEvent(self, event) -> None:  # noqa: N802
            if event.button() == Qt.LeftButton:
                pt = self._widget_to_video(event.position().toPoint())
                if pt is not None:
                    self._drawing = True
                    self.stroke_started.emit(pt[0], pt[1])

        def mouseMoveEvent(self, event) -> None:  # noqa: N802
            self._cursor_pos = event.position().toPoint()
            if self._drawing:
                pt = self._widget_to_video(self._cursor_pos)
                if pt is not None:
                    self.stroke_moved.emit(pt[0], pt[1])
            self.update()

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802
            if event.button() == Qt.LeftButton and self._drawing:
                self._drawing = False
                self.stroke_finished.emit()

        def leaveEvent(self, event) -> None:  # noqa: N802
            self._cursor_pos = None
            self.update()

    class VideoBlurMainWindow(QMainWindow):
        MODE_GLOBAL = "Global (All Frames)"
        MODE_RANGE = "Range [In..Out]"
        MODE_KEYFRAME = "Moving Keyframe"
        MODE_SINGLE = "Single Frame"

        def __init__(self, target_path: Path):
            super().__init__()
            self.setWindowTitle("Wearable Biomechanics — Video Blur Fixer (Publication QA)")
            self.resize(1280, 820)

            if target_path.is_dir():
                self.videos_dir = target_path
                self.video_list = sorted(
                    p for p in self.videos_dir.glob("*.mp4") if not p.name.startswith(".")
                )
                if not self.video_list:
                    raise FileNotFoundError(f"No .mp4 files found in {self.videos_dir}")
                self.video_idx = 0
            else:
                self.videos_dir = target_path.parent
                self.video_list = sorted(
                    p for p in self.videos_dir.glob("*.mp4") if not p.name.startswith(".")
                )
                self.video_idx = (
                    self.video_list.index(target_path) if target_path in self.video_list else 0
                )

            self.cap: cv2.VideoCapture | None = None
            self.frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
            self.max_cache_frames = 64

            self.current_frame_idx = 0
            self.total_frames = 1
            self.fps = 30.0
            self.vid_w = 1
            self.vid_h = 1

            self.range_in = 0
            self.range_out = 0
            self.project = BlurProject()
            self.undo_stack: list[dict[str, Any]] = []
            self._active_stroke: dict[str, Any] | None = None
            self._is_playing = False
            self._export_worker: ExportWorker | None = None

            self.play_timer = QTimer(self)
            self.play_timer.timeout.connect(self._on_play_tick)

            self._build_ui()
            self._bind_shortcuts()
            self._load_video(self.video_list[self.video_idx])

        def _build_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(6)

            # --- Row 1: Video Navigation & QA Status ---
            nav_row = QHBoxLayout()
            self.btn_prev_vid = QPushButton("◀ Prev Video")
            self.btn_next_vid = QPushButton("Next Video ▶")
            self.combo_videos = QComboBox()
            self.combo_videos.setMinimumWidth(260)
            for p in self.video_list:
                self.combo_videos.addItem(p.name)
            self.lbl_status = QLabel("Status: Pending")
            self.lbl_status.setStyleSheet("font-weight: bold; padding: 2px 8px; border-radius: 4px;")

            self.btn_verify_ok = QPushButton("✔ Mark Verified OK")
            self.btn_verify_ok.setStyleSheet("background-color: #1b5e20; color: white; font-weight: bold; padding: 5px 10px;")
            self.btn_export = QPushButton("💾 Save & Export Blurred MP4")
            self.btn_export.setStyleSheet("background-color: #0d47a1; color: white; font-weight: bold; padding: 5px 12px;")

            nav_row.addWidget(self.btn_prev_vid)
            nav_row.addWidget(self.combo_videos)
            nav_row.addWidget(self.btn_next_vid)
            nav_row.addWidget(self.lbl_status)
            nav_row.addStretch()
            nav_row.addWidget(self.btn_verify_ok)
            nav_row.addWidget(self.btn_export)
            root.addLayout(nav_row)

            # --- Row 2: Brush, Layer & Blur Controls ---
            tool_row = QHBoxLayout()

            tool_row.addWidget(QLabel("Layer:"))
            self.combo_mode = QComboBox()
            self.combo_mode.addItems(
                [self.MODE_GLOBAL, self.MODE_RANGE, self.MODE_KEYFRAME, self.MODE_SINGLE]
            )
            tool_row.addWidget(self.combo_mode)

            self.btn_brush = QPushButton("🖌 Brush (B)")
            self.btn_brush.setCheckable(True)
            self.btn_brush.setChecked(True)
            self.btn_eraser = QPushButton("🧹 Eraser (E)")
            self.btn_eraser.setCheckable(True)
            tool_row.addWidget(self.btn_brush)
            tool_row.addWidget(self.btn_eraser)

            tool_row.addWidget(QLabel("Size:"))
            self.spin_radius = QSpinBox()
            self.spin_radius.setRange(5, 250)
            self.spin_radius.setValue(45)
            self.spin_radius.setSuffix(" px")
            tool_row.addWidget(self.spin_radius)

            tool_row.addWidget(QLabel("Blur:"))
            self.spin_blur = QSpinBox()
            self.spin_blur.setRange(11, 151)
            self.spin_blur.setSingleStep(4)
            self.spin_blur.setValue(55)
            tool_row.addWidget(self.spin_blur)

            self.combo_blur_type = QComboBox()
            self.combo_blur_type.addItems(["gaussian", "pixelate"])
            tool_row.addWidget(self.combo_blur_type)

            self.chk_live_preview = QCheckBox("Live Blur Preview (Tab)")
            self.chk_live_preview.setChecked(True)
            tool_row.addWidget(self.chk_live_preview)

            self.btn_undo = QPushButton("↩ Undo")
            self.btn_clear_layer = QPushButton("🗑 Clear Layer")
            tool_row.addWidget(self.btn_undo)
            tool_row.addWidget(self.btn_clear_layer)
            tool_row.addStretch()
            root.addLayout(tool_row)

            # --- Row 3: Contextual Track & Range Controls ---
            ctx_row = QHBoxLayout()
            self.btn_set_in = QPushButton("Set In (I)")
            self.btn_set_out = QPushButton("Set Out (O)")
            self.lbl_range = QLabel("Range: [0 .. 0]")

            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)

            self.combo_tracks = QComboBox()
            self.combo_tracks.setMinimumWidth(140)
            self.btn_new_track = QPushButton("+ New Track")
            self.btn_del_kf = QPushButton("Del Keyframe Here")
            self.lbl_hint = QLabel("Tip: Global = paint once for all frames | Tab = toggle mask/blur | ←/→ = step frame")
            self.lbl_hint.setStyleSheet("color: #9e9e9e;")

            ctx_row.addWidget(self.btn_set_in)
            ctx_row.addWidget(self.btn_set_out)
            ctx_row.addWidget(self.lbl_range)
            ctx_row.addWidget(sep)
            ctx_row.addWidget(QLabel("Track:"))
            ctx_row.addWidget(self.combo_tracks)
            ctx_row.addWidget(self.btn_new_track)
            ctx_row.addWidget(self.btn_del_kf)
            ctx_row.addStretch()
            ctx_row.addWidget(self.lbl_hint)
            root.addLayout(ctx_row)

            # --- Center: Video Canvas ---
            self.canvas = VideoCanvas()
            root.addWidget(self.canvas, stretch=1)

            # --- Bottom: Transport & Timeline Scrubber ---
            trans_row = QHBoxLayout()
            self.btn_play = QPushButton("▶ Play (Space)")
            self.btn_prev10 = QPushButton("⏪ -10")
            self.btn_prev1 = QPushButton("◀ -1")
            self.btn_next1 = QPushButton("+1 ▶")
            self.btn_next10 = QPushButton("+10 ⏩")

            self.slider_frame = QSlider(Qt.Horizontal)
            self.slider_frame.setRange(0, 0)

            self.lbl_frame_info = QLabel("Frame 0 / 0 | 0.00 s | 30.00 fps")
            self.lbl_frame_info.setMinimumWidth(220)

            trans_row.addWidget(self.btn_play)
            trans_row.addWidget(self.btn_prev10)
            trans_row.addWidget(self.btn_prev1)
            trans_row.addWidget(self.btn_next1)
            trans_row.addWidget(self.btn_next10)
            trans_row.addWidget(self.slider_frame, stretch=1)
            trans_row.addWidget(self.lbl_frame_info)
            root.addLayout(trans_row)

            self.setStatusBar(QStatusBar())

            # Signals
            self.btn_prev_vid.clicked.connect(lambda: self._switch_video(-1))
            self.btn_next_vid.clicked.connect(lambda: self._switch_video(1))
            self.combo_videos.currentIndexChanged.connect(self._on_video_combo_changed)
            self.btn_verify_ok.clicked.connect(self._on_mark_verified_ok)
            self.btn_export.clicked.connect(self._on_export_clicked)

            self.btn_brush.clicked.connect(lambda: self._set_tool(eraser=False))
            self.btn_eraser.clicked.connect(lambda: self._set_tool(eraser=True))
            self.spin_radius.valueChanged.connect(self._sync_brush_ui)
            self.spin_blur.valueChanged.connect(self._on_blur_params_changed)
            self.combo_blur_type.currentTextChanged.connect(self._on_blur_params_changed)
            self.chk_live_preview.toggled.connect(self._refresh_canvas)
            self.btn_undo.clicked.connect(self._on_undo)
            self.btn_clear_layer.clicked.connect(self._on_clear_layer)

            self.btn_set_in.clicked.connect(self._on_set_in)
            self.btn_set_out.clicked.connect(self._on_set_out)
            self.btn_new_track.clicked.connect(self._on_new_track)
            self.btn_del_kf.clicked.connect(self._on_delete_keyframe)
            self.combo_tracks.currentIndexChanged.connect(self._refresh_canvas)

            self.btn_play.clicked.connect(self._toggle_play)
            self.btn_prev10.clicked.connect(lambda: self._step_frame(-10))
            self.btn_prev1.clicked.connect(lambda: self._step_frame(-1))
            self.btn_next1.clicked.connect(lambda: self._step_frame(1))
            self.btn_next10.clicked.connect(lambda: self._step_frame(10))
            self.slider_frame.valueChanged.connect(self._on_slider_changed)

            self.canvas.stroke_started.connect(self._on_stroke_started)
            self.canvas.stroke_moved.connect(self._on_stroke_moved)
            self.canvas.stroke_finished.connect(self._on_stroke_finished)

        def _bind_shortcuts(self) -> None:
            QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
            QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._step_frame(-1))
            QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._step_frame(1))
            QShortcut(QKeySequence("Shift+Left"), self, lambda: self._step_frame(-10))
            QShortcut(QKeySequence("Shift+Right"), self, lambda: self._step_frame(10))
            QShortcut(QKeySequence("B"), self, lambda: self._set_tool(eraser=False))
            QShortcut(QKeySequence("E"), self, lambda: self._set_tool(eraser=True))
            QShortcut(QKeySequence("["), self, lambda: self.spin_radius.setValue(max(5, self.spin_radius.value() - 8)))
            QShortcut(QKeySequence("]"), self, lambda: self.spin_radius.setValue(min(250, self.spin_radius.value() + 8)))
            QShortcut(QKeySequence("I"), self, self._on_set_in)
            QShortcut(QKeySequence("O"), self, self._on_set_out)
            QShortcut(QKeySequence(Qt.Key_Tab), self, lambda: self.chk_live_preview.toggle())
            QShortcut(QKeySequence.Undo, self, self._on_undo)

        # ---- Video loading & caching ----

        def _load_video(self, video_path: Path) -> None:
            self._stop_play()
            if self.cap is not None:
                self.cap.release()
            self.frame_cache.clear()

            # Read from backup if present so preview matches pristine un-recompressed source
            bpath = video_path.parent / BACKUP_DIR_NAME / video_path.name
            read_path = bpath if bpath.exists() else video_path

            info = probe_video(read_path)
            self.current_video_path = video_path
            self.total_frames = max(1, info.nb_frames)
            self.fps = info.fps or 30.0
            self.vid_w = info.width
            self.vid_h = info.height
            self.cap = cv2.VideoCapture(str(read_path))

            self.project = load_recipe(video_path)
            if not self.project.keyframe_tracks:
                self.project.keyframe_tracks.append({"id": 1, "name": "Track 1", "keyframes": {}})
            self.undo_stack.clear()

            self.spin_blur.blockSignals(True)
            self.spin_blur.setValue(self.project.blur_strength)
            self.spin_blur.blockSignals(False)
            self.combo_blur_type.blockSignals(True)
            self.combo_blur_type.setCurrentText(self.project.blur_mode)
            self.combo_blur_type.blockSignals(False)

            self._refresh_track_combo()
            self.range_in = 0
            self.range_out = self.total_frames - 1
            self.lbl_range.setText(f"Range: [{self.range_in} .. {self.range_out}]")

            self.slider_frame.blockSignals(True)
            self.slider_frame.setRange(0, self.total_frames - 1)
            self.slider_frame.setValue(0)
            self.slider_frame.blockSignals(False)

            self.current_frame_idx = 0
            self._sync_brush_ui()
            self._refresh_status_badge()
            self._refresh_canvas()
            self.statusBar().showMessage(
                f"Loaded {video_path.name} ({self.vid_w}x{self.vid_h}, {self.total_frames} frames @ {self.fps:.2f} fps)"
            )

        def _get_frame_bgr(self, frame_idx: int) -> np.ndarray | None:
            frame_idx = max(0, min(self.total_frames - 1, frame_idx))
            if frame_idx in self.frame_cache:
                self.frame_cache.move_to_end(frame_idx)
                return self.frame_cache[frame_idx]
            if self.cap is None:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.frame_cache[frame_idx] = frame
            if len(self.frame_cache) > self.max_cache_frames:
                self.frame_cache.popitem(last=False)
            return frame

        # ---- Rendering ----

        def _refresh_canvas(self) -> None:
            frame = self._get_frame_bgr(self.current_frame_idx)
            if frame is None:
                return
            mask = render_mask_for_frame(self.project, self.current_frame_idx, self.vid_w, self.vid_h)
            if self.chk_live_preview.isChecked():
                vis_bgr = apply_blur_to_frame(
                    frame,
                    mask,
                    blur_strength=self.project.blur_strength,
                    feather_px=self.project.feather_px,
                    mode=self.project.blur_mode,
                )
            else:
                vis_bgr = frame.copy()
                if np.any(mask):
                    overlay = vis_bgr.copy()
                    overlay[mask > 0] = (0, 210, 90)
                    vis_bgr = cv2.addWeighted(vis_bgr, 0.65, overlay, 0.35, 0)

            rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            kf_circles: list[tuple[float, float, float, bool, str]] = []
            for track in self.project.keyframe_tracks:
                interp = interpolate_track_at_frame(track, self.current_frame_idx)
                if interp is not None:
                    cx, cy, r = interp
                    is_exact = str(self.current_frame_idx) in track.get("keyframes", {})
                    kf_circles.append((cx, cy, r, is_exact, track.get("name", "Track")))

            self.canvas.set_frame_rgb(rgb, kf_circles=kf_circles)
            t_sec = self.current_frame_idx / self.fps
            self.lbl_frame_info.setText(
                f"Frame {self.current_frame_idx} / {self.total_frames - 1} | {t_sec:.2f} s | {self.fps:.2f} fps"
            )

        def _refresh_status_badge(self) -> None:
            state = load_qa_state(self.videos_dir)
            st = state.get(self.current_video_path.name, {}).get("status", "Pending")
            colors = {
                "Verified OK": "background-color: #1b5e20; color: white;",
                "Fixed": "background-color: #0d47a1; color: white;",
                "Pending": "background-color: #e65100; color: white;",
            }
            self.lbl_status.setText(f"Status: {st}")
            self.lbl_status.setStyleSheet(
                f"font-weight: bold; padding: 3px 8px; border-radius: 4px; {colors.get(st, '')}"
            )

        def _refresh_track_combo(self) -> None:
            self.combo_tracks.blockSignals(True)
            self.combo_tracks.clear()
            for t in self.project.keyframe_tracks:
                self.combo_tracks.addItem(t["name"])
            self.combo_tracks.blockSignals(False)

        # ---- Editing / Strokes ----

        def _push_undo(self) -> None:
            self.undo_stack.append(copy.deepcopy(self.project.to_dict()))
            if len(self.undo_stack) > 50:
                self.undo_stack.pop(0)

        def _on_undo(self) -> None:
            if not self.undo_stack:
                return
            self.project = BlurProject.from_dict(self.undo_stack.pop())
            self._refresh_track_combo()
            self._refresh_canvas()

        def _on_clear_layer(self) -> None:
            mode = self.combo_mode.currentText()
            self._push_undo()
            if mode == self.MODE_GLOBAL:
                self.project.global_strokes.clear()
            elif mode == self.MODE_RANGE:
                self.project.range_strokes.clear()
            elif mode == self.MODE_KEYFRAME:
                idx = self.combo_tracks.currentIndex()
                if 0 <= idx < len(self.project.keyframe_tracks):
                    self.project.keyframe_tracks[idx]["keyframes"].clear()
            elif mode == self.MODE_SINGLE:
                self.project.frame_strokes.pop(str(self.current_frame_idx), None)
            self._refresh_canvas()

        def _set_tool(self, eraser: bool) -> None:
            self.btn_brush.setChecked(not eraser)
            self.btn_eraser.setChecked(eraser)
            self._sync_brush_ui()

        def _sync_brush_ui(self) -> None:
            self.canvas.set_brush_state(self.spin_radius.value(), self.btn_eraser.isChecked())

        def _on_blur_params_changed(self) -> None:
            self.project.blur_strength = int(self.spin_blur.value()) | 1
            self.project.blur_mode = self.combo_blur_type.currentText()
            self._refresh_canvas()

        def _on_set_in(self) -> None:
            self.range_in = self.current_frame_idx
            if self.range_out < self.range_in:
                self.range_out = self.range_in
            self.lbl_range.setText(f"Range: [{self.range_in} .. {self.range_out}]")

        def _on_set_out(self) -> None:
            self.range_out = self.current_frame_idx
            if self.range_in > self.range_out:
                self.range_in = self.range_out
            self.lbl_range.setText(f"Range: [{self.range_in} .. {self.range_out}]")

        def _on_new_track(self) -> None:
            new_id = len(self.project.keyframe_tracks) + 1
            self.project.keyframe_tracks.append({"id": new_id, "name": f"Track {new_id}", "keyframes": {}})
            self._refresh_track_combo()
            self.combo_tracks.setCurrentIndex(len(self.project.keyframe_tracks) - 1)

        def _on_delete_keyframe(self) -> None:
            idx = self.combo_tracks.currentIndex()
            if 0 <= idx < len(self.project.keyframe_tracks):
                kfs = self.project.keyframe_tracks[idx]["keyframes"]
                if str(self.current_frame_idx) in kfs:
                    self._push_undo()
                    del kfs[str(self.current_frame_idx)]
                    self._refresh_canvas()

        def _on_stroke_started(self, vx: float, vy: float) -> None:
            self._stop_play()
            self._push_undo()
            mode = self.combo_mode.currentText()
            radius = self.spin_radius.value()
            erase = self.btn_eraser.isChecked()

            if mode == self.MODE_KEYFRAME:
                idx = self.combo_tracks.currentIndex()
                if 0 <= idx < len(self.project.keyframe_tracks):
                    self.project.keyframe_tracks[idx]["keyframes"][str(self.current_frame_idx)] = [
                        round(vx, 1),
                        round(vy, 1),
                        float(radius),
                    ]
                    self._refresh_canvas()
                return

            stroke: dict[str, Any] = {"points": [[round(vx, 1), round(vy, 1)]], "radius": radius, "erase": erase}
            if mode == self.MODE_GLOBAL:
                self.project.global_strokes.append(stroke)
            elif mode == self.MODE_RANGE:
                stroke["start"] = min(self.range_in, self.range_out)
                stroke["end"] = max(self.range_in, self.range_out)
                self.project.range_strokes.append(stroke)
            elif mode == self.MODE_SINGLE:
                self.project.frame_strokes.setdefault(str(self.current_frame_idx), []).append(stroke)
            self._active_stroke = stroke
            self._refresh_canvas()

        def _on_stroke_moved(self, vx: float, vy: float) -> None:
            mode = self.combo_mode.currentText()
            if mode == self.MODE_KEYFRAME:
                idx = self.combo_tracks.currentIndex()
                if 0 <= idx < len(self.project.keyframe_tracks):
                    self.project.keyframe_tracks[idx]["keyframes"][str(self.current_frame_idx)] = [
                        round(vx, 1),
                        round(vy, 1),
                        float(self.spin_radius.value()),
                    ]
                    self._refresh_canvas()
                return
            if self._active_stroke is not None:
                self._active_stroke["points"].append([round(vx, 1), round(vy, 1)])
                self._refresh_canvas()

        def _on_stroke_finished(self) -> None:
            self._active_stroke = None
            save_recipe(self.current_video_path, self.project)

        # ---- Playback & Navigation ----

        def _toggle_play(self) -> None:
            if self._is_playing:
                self._stop_play()
            else:
                self._is_playing = True
                self.btn_play.setText("⏸ Pause (Space)")
                interval_ms = max(10, int(round(1000.0 / self.fps)))
                self.play_timer.start(interval_ms)

        def _stop_play(self) -> None:
            self._is_playing = False
            self.play_timer.stop()
            self.btn_play.setText("▶ Play (Space)")

        def _on_play_tick(self) -> None:
            if self.current_frame_idx + 1 >= self.total_frames:
                self._stop_play()
                return
            self._set_frame_idx(self.current_frame_idx + 1)

        def _step_frame(self, delta: int) -> None:
            self._stop_play()
            self._set_frame_idx(self.current_frame_idx + delta)

        def _set_frame_idx(self, idx: int) -> None:
            idx = max(0, min(self.total_frames - 1, idx))
            if idx == self.current_frame_idx:
                return
            self.current_frame_idx = idx
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(idx)
            self.slider_frame.blockSignals(False)
            self._refresh_canvas()

        def _on_slider_changed(self, val: int) -> None:
            self._stop_play()
            self.current_frame_idx = val
            self._refresh_canvas()

        def _switch_video(self, delta: int) -> None:
            new_idx = max(0, min(len(self.video_list) - 1, self.video_idx + delta))
            if new_idx != self.video_idx:
                self.video_idx = new_idx
                self.combo_videos.blockSignals(True)
                self.combo_videos.setCurrentIndex(new_idx)
                self.combo_videos.blockSignals(False)
                self._load_video(self.video_list[new_idx])

        def _on_video_combo_changed(self, idx: int) -> None:
            if 0 <= idx < len(self.video_list) and idx != self.video_idx:
                self.video_idx = idx
                self._load_video(self.video_list[idx])

        # ---- QA & Export ----

        def _on_mark_verified_ok(self) -> None:
            set_video_qa_status(self.videos_dir, self.current_video_path.name, "Verified OK")
            self._refresh_status_badge()
            self.statusBar().showMessage(f"Marked {self.current_video_path.name} as Verified OK.")

        def _on_export_clicked(self) -> None:
            self._stop_play()
            if not self.project.has_any_edits():
                reply = QMessageBox.question(
                    self,
                    "No Blur Edits",
                    "No brush/keyframe edits exist for this video. Mark as 'Verified OK' instead?",
                )
                if reply == QMessageBox.Yes:
                    self._on_mark_verified_ok()
                return

            progress = QProgressDialog(
                f"Exporting {self.current_video_path.name} via FFmpeg...",
                None,
                0,
                self.total_frames,
                self,
            )
            progress.setWindowTitle("Exporting Blurred Video")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()

            worker = ExportWorker(self.current_video_path, copy.deepcopy(self.project))
            self._export_worker = worker

            def _on_prog(cur: int, tot: int) -> None:
                progress.setMaximum(max(1, tot))
                progress.setValue(cur)

            def _on_done(res: ExportResult) -> None:
                progress.close()
                set_video_qa_status(
                    self.videos_dir,
                    self.current_video_path.name,
                    "Fixed",
                    note=f"frames={res.frames_out}, fps={res.fps_out:.2f}",
                )
                self._refresh_status_badge()
                QMessageBox.information(
                    self,
                    "Export Verified ✔",
                    f"Saved: {res.video_path.name}\n"
                    f"Frames: {res.frames_out} == {res.frames_in} (EXACT MATCH)\n"
                    f"Codec: {res.codec_out} ({res.pix_fmt_out}) @ {res.fps_out:.2f} fps\n"
                    f"Original backup: {res.backup_path.name}",
                )

            def _on_fail(err: str) -> None:
                progress.close()
                QMessageBox.critical(self, "Export Failed", err)

            worker.progress.connect(_on_prog)
            worker.finished_ok.connect(_on_done)
            worker.failed.connect(_on_fail)
            worker.start()

        def closeEvent(self, event) -> None:  # noqa: N802
            self._stop_play()
            if self.cap is not None:
                self.cap.release()
            super().closeEvent(event)

    return QApplication, VideoBlurMainWindow


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m src.export.video_blur_ui <path_to_videos_dir_or_mp4>")
        return 1
    target = Path(args[0]).expanduser().resolve()
    QApplication, VideoBlurMainWindow = _build_gui_classes()
    app = QApplication.instance() or QApplication(sys.argv)
    win = VideoBlurMainWindow(target)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
