"""Inline HTML5 Canvas Video Blur Editor for Google Colab (Mode A).

Uses `google.colab.output.register_callback` to bridge a 60-fps client-side
HTML5 brush/keyframe canvas directly to `src.export.video_blur_core`.
"""

from __future__ import annotations

import base64
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.export.video_blur_core import (
    BACKUP_DIR_NAME,
    BlurProject,
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

# Global active session for Colab callbacks
_ACTIVE_SESSION: "_ColabBlurSession | None" = None


class _ColabBlurSession:
    """Stateful video & recipe session backing the inline HTML5 canvas UI."""

    def __init__(self, target_path: str | Path):
        p = Path(target_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

        if p.is_dir():
            self.videos_dir = p
            self.video_list = sorted(x for x in self.videos_dir.glob("*.mp4") if not x.name.startswith("."))
            if not self.video_list:
                raise FileNotFoundError(f"No .mp4 files found in {self.videos_dir}")
            self.video_idx = 0
        else:
            self.videos_dir = p.parent
            self.video_list = sorted(x for x in self.videos_dir.glob("*.mp4") if not x.name.startswith("."))
            self.video_idx = self.video_list.index(p) if p in self.video_list else 0

        self.cap: cv2.VideoCapture | None = None
        self.frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_cache_frames = 48

        self.current_video_path: Path = self.video_list[self.video_idx]
        self.total_frames: int = 1
        self.fps: float = 30.0
        self.vid_w: int = 1
        self.vid_h: int = 1
        self.project: BlurProject = BlurProject()
        self.undo_stack: list[dict[str, Any]] = []

        self.load_video(self.video_idx)

    def load_video(self, idx: int) -> dict[str, Any]:
        idx = max(0, min(len(self.video_list) - 1, int(idx)))
        self.video_idx = idx
        video_path = self.video_list[idx]

        if self.cap is not None:
            self.cap.release()
        self.frame_cache.clear()

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

        return self.get_state_summary()

    def get_state_summary(self) -> dict[str, Any]:
        qa = load_qa_state(self.videos_dir)
        status = qa.get(self.current_video_path.name, {}).get("status", "Pending")
        return {
            "video_idx": self.video_idx,
            "video_name": self.current_video_path.name,
            "videos": [p.name for p in self.video_list],
            "status": status,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 3),
            "width": self.vid_w,
            "height": self.vid_h,
            "blur_strength": self.project.blur_strength,
            "blur_mode": self.project.blur_mode,
            "tracks": [{"id": t["id"], "name": t["name"]} for t in self.project.keyframe_tracks],
        }

    def _get_frame_bgr(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, min(self.total_frames - 1, int(frame_idx)))
        if frame_idx in self.frame_cache:
            self.frame_cache.move_to_end(frame_idx)
            return self.frame_cache[frame_idx]
        if self.cap is None:
            raise RuntimeError("VideoCapture not initialized.")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed reading frame {frame_idx} from {self.current_video_path.name}")
        self.frame_cache[frame_idx] = frame
        if len(self.frame_cache) > self.max_cache_frames:
            self.frame_cache.popitem(last=False)
        return frame

    def get_frame_payload(self, frame_idx: int, live_preview: bool = True, jpeg_quality: int = 85) -> dict[str, Any]:
        frame_idx = max(0, min(self.total_frames - 1, int(frame_idx)))
        frame = self._get_frame_bgr(frame_idx)
        mask = render_mask_for_frame(self.project, frame_idx, self.vid_w, self.vid_h)

        if live_preview:
            vis = apply_blur_to_frame(
                frame,
                mask,
                blur_strength=self.project.blur_strength,
                feather_px=self.project.feather_px,
                mode=self.project.blur_mode,
            )
        else:
            vis = frame.copy()
            if np.any(mask):
                overlay = vis.copy()
                overlay[mask > 0] = (0, 210, 90)
                vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("JPEG encode failed.")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        kf_guides = []
        for track in self.project.keyframe_tracks:
            interp = interpolate_track_at_frame(track, frame_idx)
            if interp is not None:
                cx, cy, r = interp
                kf_guides.append(
                    {
                        "cx": cx,
                        "cy": cy,
                        "r": r,
                        "exact": str(frame_idx) in track.get("keyframes", {}),
                        "name": track.get("name", "Track"),
                    }
                )

        return {
            "frame_idx": frame_idx,
            "img": f"data:image/jpeg;base64,{b64}",
            "kf_guides": kf_guides,
        }

    def _push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.project.to_dict()))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def add_stroke(self, mode: str, stroke: dict[str, Any], frame_idx: int, track_idx: int = 0) -> None:
        self._push_undo()
        if mode == "global":
            self.project.global_strokes.append(stroke)
        elif mode == "range":
            self.project.range_strokes.append(stroke)
        elif mode == "single":
            self.project.frame_strokes.setdefault(str(int(frame_idx)), []).append(stroke)
        elif mode == "keyframe":
            if 0 <= int(track_idx) < len(self.project.keyframe_tracks):
                pts = stroke.get("points", [])
                if pts:
                    cx, cy = pts[-1]
                    r = float(stroke.get("radius", 40))
                    self.project.keyframe_tracks[int(track_idx)]["keyframes"][str(int(frame_idx))] = [
                        round(float(cx), 1),
                        round(float(cy), 1),
                        r,
                    ]
        save_recipe(self.current_video_path, self.project)

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.project = BlurProject.from_dict(self.undo_stack.pop())
        save_recipe(self.current_video_path, self.project)

    def clear_layer(self, mode: str, frame_idx: int, track_idx: int = 0) -> None:
        self._push_undo()
        if mode == "global":
            self.project.global_strokes.clear()
        elif mode == "range":
            self.project.range_strokes.clear()
        elif mode == "single":
            self.project.frame_strokes.pop(str(int(frame_idx)), None)
        elif mode == "keyframe":
            if 0 <= int(track_idx) < len(self.project.keyframe_tracks):
                self.project.keyframe_tracks[int(track_idx)]["keyframes"].clear()
        save_recipe(self.current_video_path, self.project)

    def new_track(self) -> list[dict[str, Any]]:
        new_id = len(self.project.keyframe_tracks) + 1
        self.project.keyframe_tracks.append({"id": new_id, "name": f"Track {new_id}", "keyframes": {}})
        save_recipe(self.current_video_path, self.project)
        return [{"id": t["id"], "name": t["name"]} for t in self.project.keyframe_tracks]

    def delete_keyframe(self, frame_idx: int, track_idx: int = 0) -> None:
        if 0 <= int(track_idx) < len(self.project.keyframe_tracks):
            kfs = self.project.keyframe_tracks[int(track_idx)]["keyframes"]
            if str(int(frame_idx)) in kfs:
                self._push_undo()
                del kfs[str(int(frame_idx))]
                save_recipe(self.current_video_path, self.project)

    def set_blur_params(self, strength: int, mode: str) -> None:
        self.project.blur_strength = max(11, int(strength) | 1)
        self.project.blur_mode = "pixelate" if mode == "pixelate" else "gaussian"
        save_recipe(self.current_video_path, self.project)

    def mark_verified_ok(self) -> dict[str, Any]:
        set_video_qa_status(self.videos_dir, self.current_video_path.name, "Verified OK")
        return self.get_state_summary()

    def export_current(self) -> dict[str, Any]:
        if not self.project.has_any_edits():
            return {"ok": False, "message": "No brush/keyframe edits exist. Use 'Mark Verified OK' if clean."}
        res = export_blurred_video(self.current_video_path, self.project)
        set_video_qa_status(
            self.videos_dir,
            self.current_video_path.name,
            "Fixed",
            note=f"frames={res.frames_out}, fps={res.fps_out:.2f}",
        )
        return {
            "ok": True,
            "message": (
                f"Saved {res.video_path.name} | Frames: {res.frames_out} == {res.frames_in} (EXACT MATCH) | "
                f"{res.codec_out} ({res.pix_fmt_out}) @ {res.fps_out:.2f} fps | Backup: {res.backup_path.name}"
            ),
            "state": self.get_state_summary(),
        }


def _register_colab_callbacks() -> None:
    from google.colab import output as colab_output
    from IPython.display import JSON

    def _require_session() -> _ColabBlurSession:
        if _ACTIVE_SESSION is None:
            raise RuntimeError("No active blur session. Re-run launch_colab_blur_editor().")
        return _ACTIVE_SESSION

    colab_output.register_callback(
        "vbf.load_video",
        lambda idx: JSON(_require_session().load_video(int(idx))),
    )
    colab_output.register_callback(
        "vbf.get_frame",
        lambda frame_idx, live_preview: JSON(
            _require_session().get_frame_payload(int(frame_idx), bool(live_preview))
        ),
    )
    colab_output.register_callback(
        "vbf.add_stroke",
        lambda mode, stroke_json, frame_idx, track_idx, live_preview: (
            _require_session().add_stroke(str(mode), json.loads(stroke_json), int(frame_idx), int(track_idx)),
            JSON(_require_session().get_frame_payload(int(frame_idx), bool(live_preview))),
        )[1],
    )
    colab_output.register_callback(
        "vbf.undo",
        lambda frame_idx, live_preview: (
            _require_session().undo(),
            JSON(_require_session().get_frame_payload(int(frame_idx), bool(live_preview))),
        )[1],
    )
    colab_output.register_callback(
        "vbf.clear_layer",
        lambda mode, frame_idx, track_idx, live_preview: (
            _require_session().clear_layer(str(mode), int(frame_idx), int(track_idx)),
            JSON(_require_session().get_frame_payload(int(frame_idx), bool(live_preview))),
        )[1],
    )
    colab_output.register_callback(
        "vbf.new_track",
        lambda: JSON({"tracks": _require_session().new_track()}),
    )
    colab_output.register_callback(
        "vbf.del_kf",
        lambda frame_idx, track_idx, live_preview: (
            _require_session().delete_keyframe(int(frame_idx), int(track_idx)),
            JSON(_require_session().get_frame_payload(int(frame_idx), bool(live_preview))),
        )[1],
    )
    colab_output.register_callback(
        "vbf.set_blur",
        lambda strength, mode, frame_idx, live_preview: (
            _require_session().set_blur_params(int(strength), str(mode)),
            JSON(_require_session().get_frame_payload(int(frame_idx), bool(live_preview))),
        )[1],
    )
    colab_output.register_callback(
        "vbf.mark_ok",
        lambda: JSON(_require_session().mark_verified_ok()),
    )
    colab_output.register_callback(
        "vbf.export",
        lambda: JSON(_require_session().export_current()),
    )


def _build_html_widget(init_state: dict[str, Any], init_frame: dict[str, Any]) -> str:
    init_state_json = json.dumps(init_state)
    init_frame_json = json.dumps(init_frame)
    return f"""
<div id="vbf-root" style="font-family: system-ui, -apple-system, sans-serif; background:#14171f; color:#e6edf3; padding:12px; border-radius:10px; border:1px solid #30363d; max-width:1180px;">
  <style>
    #vbf-root button {{ background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:5px 10px; cursor:pointer; font-size:12px; }}
    #vbf-root button:hover {{ background:#30363d; }}
    #vbf-root button.active {{ background:#238636; border-color:#2ea043; color:#fff; }}
    #vbf-root button.eraser-active {{ background:#b62324; border-color:#f85149; color:#fff; }}
    #vbf-root button.primary {{ background:#1f6feb; border-color:#388bfd; color:#fff; font-weight:600; }}
    #vbf-root button.success {{ background:#238636; border-color:#2ea043; color:#fff; font-weight:600; }}
    #vbf-root select, #vbf-root input[type="number"] {{ background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:4px 6px; font-size:12px; }}
    .vbf-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:8px; }}
    .vbf-badge {{ padding:3px 9px; border-radius:999px; font-size:12px; font-weight:600; }}
    .vbf-badge.Pending {{ background:#9e6a03; color:#fff; }}
    .vbf-badge.Fixed {{ background:#1f6feb; color:#fff; }}
    .vbf-badge.Verified {{ background:#238636; color:#fff; }}
  </style>

  <!-- Row 1: Video nav & QA actions -->
  <div class="vbf-row">
    <button id="vbf-prev-vid">◀ Prev Video</button>
    <select id="vbf-vid-select" style="min-width:240px;"></select>
    <button id="vbf-next-vid">Next Video ▶</button>
    <span id="vbf-status-badge" class="vbf-badge Pending">Pending</span>
    <span style="flex:1;"></span>
    <button id="vbf-mark-ok" class="success">✔ Mark Verified OK</button>
    <button id="vbf-export" class="primary">💾 Save &amp; Export Blurred MP4</button>
  </div>

  <!-- Row 2: Layer & brush controls -->
  <div class="vbf-row">
    <label style="font-size:12px;">Layer:</label>
    <select id="vbf-mode">
      <option value="global">Global (All Frames)</option>
      <option value="range">Range [In..Out]</option>
      <option value="keyframe">Moving Keyframe</option>
      <option value="single">Single Frame</option>
    </select>
    <button id="vbf-tool-brush" class="active">🖌 Brush (B)</button>
    <button id="vbf-tool-eraser">🧹 Eraser (E)</button>
    <label style="font-size:12px;">Size:</label>
    <input id="vbf-radius" type="number" min="5" max="250" value="45" style="width:62px;" />
    <label style="font-size:12px;">Blur:</label>
    <input id="vbf-blur" type="number" min="11" max="151" step="4" value="55" style="width:62px;" />
    <select id="vbf-blur-mode">
      <option value="gaussian">gaussian</option>
      <option value="pixelate">pixelate</option>
    </select>
    <label style="font-size:12px; display:flex; align-items:center; gap:4px; cursor:pointer;">
      <input id="vbf-live" type="checkbox" checked /> Live Blur Preview (Tab)
    </label>
    <button id="vbf-undo">↩ Undo</button>
    <button id="vbf-clear">🗑 Clear Layer</button>
  </div>

  <!-- Row 3: Range & Track controls -->
  <div class="vbf-row" style="font-size:12px; color:#8b949e;">
    <button id="vbf-set-in">Set In (I)</button>
    <button id="vbf-set-out">Set Out (O)</button>
    <span id="vbf-range-lbl">Range: [0 .. 0]</span>
    <span>|</span>
    <label>Track:</label>
    <select id="vbf-track-select"></select>
    <button id="vbf-new-track">+ New Track</button>
    <button id="vbf-del-kf">Del Keyframe Here</button>
    <span style="flex:1;"></span>
    <span>Shortcuts: ←/→ step | Shift+←/→ ±10 | [ / ] size | Tab preview</span>
  </div>

  <!-- Canvas Container -->
  <div style="position:relative; background:#0b0d12; border:1px solid #30363d; border-radius:6px; overflow:hidden; display:flex; justify-content:center; align-items:center; min-height:360px;">
    <canvas id="vbf-canvas" style="display:block; max-width:100%; max-height:68vh; cursor:crosshair;"></canvas>
  </div>

  <!-- Transport -->
  <div class="vbf-row" style="margin-top:8px; margin-bottom:4px;">
    <button id="vbf-play">▶ Play</button>
    <button id="vbf-m10">⏪ -10</button>
    <button id="vbf-m1">◀ -1</button>
    <button id="vbf-p1">+1 ▶</button>
    <button id="vbf-p10">+10 ⏩</button>
    <input id="vbf-slider" type="range" min="0" max="0" value="0" style="flex:1;" />
    <span id="vbf-frame-lbl" style="font-size:12px; min-width:190px; text-align:right;">Frame 0 / 0</span>
  </div>

  <div id="vbf-msg" style="font-size:12px; color:#58a6ff; min-height:18px; margin-top:4px;"></div>
</div>

<script>
(() => {{
  const root = document.getElementById("vbf-root");
  const canvas = document.getElementById("vbf-canvas");
  const ctx = canvas.getContext("2d");

  const elVidSel = document.getElementById("vbf-vid-select");
  const elStatus = document.getElementById("vbf-status-badge");
  const elMode = document.getElementById("vbf-mode");
  const elBrush = document.getElementById("vbf-tool-brush");
  const elEraser = document.getElementById("vbf-tool-eraser");
  const elRadius = document.getElementById("vbf-radius");
  const elBlur = document.getElementById("vbf-blur");
  const elBlurMode = document.getElementById("vbf-blur-mode");
  const elLive = document.getElementById("vbf-live");
  const elRangeLbl = document.getElementById("vbf-range-lbl");
  const elTrackSel = document.getElementById("vbf-track-select");
  const elSlider = document.getElementById("vbf-slider");
  const elFrameLbl = document.getElementById("vbf-frame-lbl");
  const elMsg = document.getElementById("vbf-msg");
  const elPlay = document.getElementById("vbf-play");

  let state = {init_state_json};
  let curFrame = 0;
  let rangeIn = 0;
  let rangeOut = Math.max(0, state.total_frames - 1);
  let isEraser = false;
  let drawing = false;
  let activePts = [];
  let kfGuides = [];
  let bgImg = new Image();
  let hoverPos = null;
  let playing = false;
  let playTimer = null;
  let busy = false;

  async function invoke(name, args) {{
    const res = await google.colab.kernel.invokeFunction(name, args, {{}});
    return res.data["application/json"];
  }}

  function setMsg(txt, isErr=false) {{
    elMsg.style.color = isErr ? "#f85149" : "#58a6ff";
    elMsg.textContent = txt;
  }}

  function updateStatusBadge(st) {{
    elStatus.textContent = st;
    elStatus.className = "vbf-badge " + (st.startsWith("Verified") ? "Verified" : st);
  }}

  function populateTracks(tracks) {{
    const prev = elTrackSel.value;
    elTrackSel.innerHTML = "";
    tracks.forEach((t, i) => {{
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = t.name;
      elTrackSel.appendChild(opt);
    }});
    if (prev && Number(prev) < tracks.length) elTrackSel.value = prev;
  }}

  function applyState(st) {{
    state = st;
    elVidSel.innerHTML = "";
    st.videos.forEach((v, i) => {{
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = v;
      if (i === st.video_idx) opt.selected = true;
      elVidSel.appendChild(opt);
    }});
    updateStatusBadge(st.status);
    elBlur.value = st.blur_strength;
    elBlurMode.value = st.blur_mode;
    populateTracks(st.tracks);
    curFrame = 0;
    rangeIn = 0;
    rangeOut = Math.max(0, st.total_frames - 1);
    elRangeLbl.textContent = `Range: [${{rangeIn}} .. ${{rangeOut}}]`;
    elSlider.max = Math.max(0, st.total_frames - 1);
    elSlider.value = 0;
    canvas.width = st.width;
    canvas.height = st.height;
  }}

  function drawCanvas() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (bgImg.complete && bgImg.naturalWidth > 0) {{
      ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
    }}
    // Keyframe guides
    kfGuides.forEach(g => {{
      ctx.save();
      ctx.beginPath();
      ctx.arc(g.cx, g.cy, g.r, 0, Math.PI * 2);
      ctx.strokeStyle = g.exact ? "#00e5ff" : "#ffd54f";
      ctx.lineWidth = 2;
      if (!g.exact) ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.fillStyle = g.exact ? "#00e5ff" : "#ffd54f";
      ctx.font = "13px sans-serif";
      ctx.fillText(g.name, g.cx - g.r, g.cy - g.r - 4);
      ctx.restore();
    }});
    // Active stroke preview (60 fps client-side)
    if (drawing && activePts.length > 0) {{
      const rad = Number(elRadius.value) || 40;
      ctx.save();
      if (elMode.value === "keyframe") {{
        const p = activePts[activePts.length - 1];
        ctx.beginPath();
        ctx.arc(p[0], p[1], rad, 0, Math.PI * 2);
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 3;
        ctx.stroke();
      }} else {{
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = rad * 2;
        ctx.strokeStyle = isEraser ? "rgba(248,81,73,0.45)" : "rgba(46,160,67,0.45)";
        ctx.beginPath();
        ctx.moveTo(activePts[0][0], activePts[0][1]);
        for (let i = 1; i < activePts.length; i++) ctx.lineTo(activePts[i][0], activePts[i][1]);
        ctx.stroke();
      }}
      ctx.restore();
    }}
    // Brush cursor
    if (hoverPos) {{
      const rad = Number(elRadius.value) || 40;
      ctx.save();
      ctx.beginPath();
      ctx.arc(hoverPos[0], hoverPos[1], rad, 0, Math.PI * 2);
      ctx.strokeStyle = isEraser ? "#f85149" : "#69f0ae";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    }}
  }}

  function renderFramePayload(payload) {{
    curFrame = payload.frame_idx;
    kfGuides = payload.kf_guides || [];
    elSlider.value = curFrame;
    const sec = (curFrame / (state.fps || 30)).toFixed(2);
    elFrameLbl.textContent = `Frame ${{curFrame}} / ${{state.total_frames - 1}} | ${{sec}}s | ${{state.fps}} fps`;
    bgImg.onload = () => drawCanvas();
    bgImg.src = payload.img;
  }}

  async function fetchFrame(idx) {{
    idx = Math.max(0, Math.min(state.total_frames - 1, idx));
    const payload = await invoke("vbf.get_frame", [idx, elLive.checked]);
    renderFramePayload(payload);
  }}

  function getVideoCoords(evt) {{
    const rect = canvas.getBoundingClientRect();
    const x = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const y = (evt.clientY - rect.top) * (canvas.height / rect.height);
    return [Math.max(0, Math.min(canvas.width - 1, x)), Math.max(0, Math.min(canvas.height - 1, y))];
  }}

  function stopPlay() {{
    playing = false;
    elPlay.textContent = "▶ Play";
    if (playTimer) {{ clearTimeout(playTimer); playTimer = null; }}
  }}

  async function playStep() {{
    if (!playing) return;
    if (curFrame + 1 >= state.total_frames) {{ stopPlay(); return; }}
    await fetchFrame(curFrame + 1);
    if (playing) playTimer = setTimeout(playStep, 60);
  }}

  // Mouse painting
  canvas.addEventListener("mousedown", (e) => {{
    if (e.button !== 0) return;
    stopPlay();
    drawing = true;
    activePts = [getVideoCoords(e)];
    drawCanvas();
  }});
  canvas.addEventListener("mousemove", (e) => {{
    const pt = getVideoCoords(e);
    hoverPos = pt;
    if (drawing) activePts.push(pt);
    drawCanvas();
  }});
  window.addEventListener("mouseup", async () => {{
    if (!drawing) return;
    drawing = false;
    const mode = elMode.value;
    const stroke = {{
      points: activePts.map(p => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]),
      radius: Number(elRadius.value) || 40,
      erase: isEraser,
    }};
    if (mode === "range") {{
      stroke.start = Math.min(rangeIn, rangeOut);
      stroke.end = Math.max(rangeIn, rangeOut);
    }}
    activePts = [];
    const payload = await invoke("vbf.add_stroke", [mode, JSON.stringify(stroke), curFrame, Number(elTrackSel.value || 0), elLive.checked]);
    renderFramePayload(payload);
  }});
  canvas.addEventListener("mouseleave", () => {{ hoverPos = null; drawCanvas(); }});

  // Buttons & controls
  elBrush.onclick = () => {{ isEraser = false; elBrush.className = "active"; elEraser.className = ""; drawCanvas(); }};
  elEraser.onclick = () => {{ isEraser = true; elBrush.className = ""; elEraser.className = "eraser-active"; drawCanvas(); }};
  elLive.onchange = () => fetchFrame(curFrame);
  elBlur.onchange = async () => renderFramePayload(await invoke("vbf.set_blur", [Number(elBlur.value), elBlurMode.value, curFrame, elLive.checked]));
  elBlurMode.onchange = async () => renderFramePayload(await invoke("vbf.set_blur", [Number(elBlur.value), elBlurMode.value, curFrame, elLive.checked]));

  document.getElementById("vbf-undo").onclick = async () => renderFramePayload(await invoke("vbf.undo", [curFrame, elLive.checked]));
  document.getElementById("vbf-clear").onclick = async () => renderFramePayload(await invoke("vbf.clear_layer", [elMode.value, curFrame, Number(elTrackSel.value || 0), elLive.checked]));
  document.getElementById("vbf-set-in").onclick = () => {{ rangeIn = curFrame; if (rangeOut < rangeIn) rangeOut = rangeIn; elRangeLbl.textContent = `Range: [${{rangeIn}} .. ${{rangeOut}}]`; }};
  document.getElementById("vbf-set-out").onclick = () => {{ rangeOut = curFrame; if (rangeIn > rangeOut) rangeIn = rangeOut; elRangeLbl.textContent = `Range: [${{rangeIn}} .. ${{rangeOut}}]`; }};
  document.getElementById("vbf-new-track").onclick = async () => {{
    const r = await invoke("vbf.new_track", []);
    populateTracks(r.tracks);
    elTrackSel.value = r.tracks.length - 1;
  }};
  document.getElementById("vbf-del-kf").onclick = async () => renderFramePayload(await invoke("vbf.del_kf", [curFrame, Number(elTrackSel.value || 0), elLive.checked]));

  elPlay.onclick = () => {{
    if (playing) {{ stopPlay(); }} else {{ playing = true; elPlay.textContent = "⏸ Pause"; playStep(); }}
  }};
  document.getElementById("vbf-m10").onclick = () => {{ stopPlay(); fetchFrame(curFrame - 10); }};
  document.getElementById("vbf-m1").onclick = () => {{ stopPlay(); fetchFrame(curFrame - 1); }};
  document.getElementById("vbf-p1").onclick = () => {{ stopPlay(); fetchFrame(curFrame + 1); }};
  document.getElementById("vbf-p10").onclick = () => {{ stopPlay(); fetchFrame(curFrame + 10); }};
  elSlider.oninput = () => {{ stopPlay(); fetchFrame(Number(elSlider.value)); }};

  async function switchVideo(newIdx) {{
    stopPlay();
    setMsg("Loading video...");
    const st = await invoke("vbf.load_video", [newIdx]);
    applyState(st);
    await fetchFrame(0);
    setMsg(`Loaded ${{st.video_name}} (${{st.width}}x${{st.height}}, ${{st.total_frames}} frames @ ${{st.fps}} fps)`);
  }}
  document.getElementById("vbf-prev-vid").onclick = () => switchVideo(Math.max(0, state.video_idx - 1));
  document.getElementById("vbf-next-vid").onclick = () => switchVideo(Math.min(state.videos.length - 1, state.video_idx + 1));
  elVidSel.onchange = () => switchVideo(Number(elVidSel.value));

  document.getElementById("vbf-mark-ok").onclick = async () => {{
    const st = await invoke("vbf.mark_ok", []);
    updateStatusBadge(st.status);
    setMsg(`Marked ${{st.video_name}} as Verified OK.`);
  }};
  document.getElementById("vbf-export").onclick = async () => {{
    if (busy) return;
    stopPlay();
    busy = true;
    setMsg("Exporting via FFmpeg & verifying exact frame count via ffprobe... (please wait)");
    try {{
      const res = await invoke("vbf.export", []);
      if (res.ok) {{
        updateStatusBadge(res.state.status);
        setMsg("✔ " + res.message);
      }} else {{
        setMsg(res.message, true);
      }}
    }} catch (err) {{
      setMsg("Export failed: " + err, true);
    }} finally {{
      busy = false;
    }}
  }};

  // Keyboard shortcuts when hovered/focused
  root.tabIndex = 0;
  root.addEventListener("keydown", (e) => {{
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.key === "ArrowLeft") {{ e.preventDefault(); stopPlay(); fetchFrame(curFrame - (e.shiftKey ? 10 : 1)); }}
    else if (e.key === "ArrowRight") {{ e.preventDefault(); stopPlay(); fetchFrame(curFrame + (e.shiftKey ? 10 : 1)); }}
    else if (e.key === "b" || e.key === "B") {{ elBrush.click(); }}
    else if (e.key === "e" || e.key === "E") {{ elEraser.click(); }}
    else if (e.key === "[") {{ elRadius.value = Math.max(5, Number(elRadius.value) - 8); drawCanvas(); }}
    else if (e.key === "]") {{ elRadius.value = Math.min(250, Number(elRadius.value) + 8); drawCanvas(); }}
    else if (e.key === "i" || e.key === "I") {{ document.getElementById("vbf-set-in").click(); }}
    else if (e.key === "o" || e.key === "O") {{ document.getElementById("vbf-set-out").click(); }}
    else if (e.key === "Tab") {{ e.preventDefault(); elLive.checked = !elLive.checked; fetchFrame(curFrame); }}
  }});

  // Initial render
  applyState(state);
  renderFramePayload({init_frame_json});
  setMsg(`Loaded ${{state.video_name}} (${{state.width}}x${{state.height}}, ${{state.total_frames}} frames @ ${{state.fps}} fps)`);
}})();
</script>
"""


def launch_colab_blur_editor(target_path: str | Path) -> None:
    """Display the inline HTML5 Canvas Video Blur Fixer inside Google Colab."""
    global _ACTIVE_SESSION
    from IPython.display import HTML, display

    _ACTIVE_SESSION = _ColabBlurSession(target_path)
    _register_colab_callbacks()
    init_state = _ACTIVE_SESSION.get_state_summary()
    init_frame = _ACTIVE_SESSION.get_frame_payload(0, live_preview=True)
    display(HTML(_build_html_widget(init_state, init_frame)))
