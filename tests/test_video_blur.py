"""Unit and integration tests for post-export video blur fixer backend & PySide6 UI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.export.video_blur_core import (
    BACKUP_DIR_NAME,
    BlurProject,
    apply_blur_to_frame,
    export_blurred_video,
    interpolate_track_at_frame,
    probe_video,
    render_mask_for_frame,
    scan_videos_folder,
    set_video_qa_status,
    update_checksums_sha256,
    verify_all_frame_counts,
)
from src.export.video_blur_colab import _ColabBlurSession, _build_html_widget
from src.export.video_blur_ui import _build_gui_classes


def _make_test_video(path: Path, n_frames: int = 30, w: int = 160, h: int = 120, fps: int = 30) -> None:
    """Create a deterministic high-frequency checkerboard H.264 MP4 via FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // 4) + (yy // 4)) % 2 * 255).astype(np.uint8)
    for f in range(n_frames):
        frame = np.stack([checker, np.roll(checker, f, axis=1), checker], axis=-1)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    assert ret == 0, proc.stderr.read().decode("utf-8", errors="replace")


def test_mask_layers_and_interpolation() -> None:
    proj = BlurProject(
        global_strokes=[{"points": [[20, 20]], "radius": 10, "erase": False}],
        range_strokes=[{"start": 5, "end": 10, "points": [[80, 20]], "radius": 10, "erase": False}],
        keyframe_tracks=[
            {
                "id": 1,
                "name": "Track 1",
                "keyframes": {"0": [20.0, 80.0, 10.0], "10": [120.0, 80.0, 20.0]},
            }
        ],
        frame_strokes={"5": [{"points": [[20, 20]], "radius": 12, "erase": True}]},
    )

    # Midpoint interpolation at frame 5 -> (70, 80, 15)
    interp = interpolate_track_at_frame(proj.keyframe_tracks[0], 5)
    assert interp is not None
    assert np.allclose(interp, (70.0, 80.0, 15.0))
    assert interpolate_track_at_frame(proj.keyframe_tracks[0], 11) is None

    # Frame 0: global active, range inactive, track at (20, 80)
    m0 = render_mask_for_frame(proj, 0, 160, 120)
    assert m0[20, 20] == 255
    assert m0[20, 80] == 0
    assert m0[80, 20] == 255

    # Frame 5: global erased by single-frame override, range active at (80, 20), track at (70, 80)
    m5 = render_mask_for_frame(proj, 5, 160, 120)
    assert m5[20, 20] == 0
    assert m5[20, 80] == 255
    assert m5[80, 70] == 255


def test_apply_blur_modes() -> None:
    yy, xx = np.mgrid[0:120, 0:160]
    checker = (((xx // 4) + (yy // 4)) % 2 * 255).astype(np.uint8)
    frame = np.stack([checker, checker, checker], axis=-1)

    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 15, 255, -1)

    for mode in ("gaussian", "pixelate"):
        out = apply_blur_to_frame(frame, mask, blur_strength=31, feather_px=9, mode=mode)
        # Inside masked region should have lower variance than sharp checkerboard
        assert out[30:50, 30:50].std() < frame[30:50, 30:50].std()
        # Far outside masked region must be bit-exact untouched
        assert np.array_equal(out[95:115, 130:155], frame[95:115, 130:155])


def test_export_audit_and_checksums(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "Clean&JerkDataset"
    videos_dir = dataset_dir / "videos"
    videos_dir.mkdir(parents=True)
    vid_path = videos_dir / "set_0001_front.mp4"
    _make_test_video(vid_path, n_frames=30, w=160, h=120, fps=30)

    # Seed checksums.sha256 before modification
    csum_init = update_checksums_sha256(dataset_dir)
    assert csum_init["added"] == 1

    proj = BlurProject(
        blur_strength=35,
        feather_px=11,
        global_strokes=[{"points": [[40, 40], [60, 40]], "radius": 18, "erase": False}],
    )
    res = export_blurred_video(vid_path, proj)
    assert res.frames_in == 30
    assert res.frames_out == 30
    assert abs(res.fps_out - 30.0) < 0.01
    assert res.codec_out == "h264"
    assert res.pix_fmt_out == "yuv420p"
    assert (videos_dir / BACKUP_DIR_NAME / vid_path.name).exists()

    set_video_qa_status(videos_dir, vid_path.name, "Fixed")
    df = scan_videos_folder(videos_dir, probe_all=True)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "Fixed"
    assert bool(df.iloc[0]["has_backup"]) is True

    audit = verify_all_frame_counts(videos_dir)
    assert len(audit) == 1
    assert bool(audit.iloc[0]["frame_match"]) is True

    csum_after = update_checksums_sha256(dataset_dir)
    assert csum_after["updated"] == 1


def test_pyside6_ui_headless(tmp_path: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    vid_path = tmp_path / "set_0002_side.mp4"
    _make_test_video(vid_path, n_frames=12, w=160, h=120, fps=30)

    QApplication, VideoBlurMainWindow = _build_gui_classes()
    app = QApplication.instance() or QApplication([])
    win = VideoBlurMainWindow(vid_path)
    assert win.total_frames == 12

    # Simulate a global brush stroke & undo
    win._on_stroke_started(40.0, 40.0)
    win._on_stroke_moved(60.0, 45.0)
    win._on_stroke_finished()
    assert len(win.project.global_strokes) == 1

    win._on_undo()
    assert len(win.project.global_strokes) == 0

    # Step frames & mark verified
    win._step_frame(3)
    assert win.current_frame_idx == 3
    win._on_mark_verified_ok()
    win.close()
    app.processEvents()


def test_colab_session_and_html(tmp_path: Path) -> None:
    vid_path = tmp_path / "set_0003_front.mp4"
    _make_test_video(vid_path, n_frames=15, w=160, h=120, fps=30)

    sess = _ColabBlurSession(vid_path)
    st = sess.get_state_summary()
    assert st["total_frames"] == 15
    assert st["video_name"] == "set_0003_front.mp4"

    p0 = sess.get_frame_payload(0, live_preview=True)
    assert p0["img"].startswith("data:image/jpeg;base64,")

    # Add global brush stroke, keyframe track, and export
    sess.add_stroke("global", {"points": [[40, 40], [60, 40]], "radius": 18, "erase": False}, frame_idx=0)
    sess.add_stroke("keyframe", {"points": [[80, 60]], "radius": 15}, frame_idx=0, track_idx=0)
    sess.add_stroke("keyframe", {"points": [[100, 60]], "radius": 15}, frame_idx=10, track_idx=0)

    p5 = sess.get_frame_payload(5, live_preview=True)
    assert len(p5["kf_guides"]) == 1
    assert abs(p5["kf_guides"][0]["cx"] - 90.0) < 1e-3

    html = _build_html_widget(st, p0)
    assert "vbf-canvas" in html
    assert "vbf.export" in html

    exp = sess.export_current()
    assert exp["ok"] is True
    assert exp["state"]["status"] == "Fixed"
    assert (tmp_path / BACKUP_DIR_NAME / vid_path.name).exists()
