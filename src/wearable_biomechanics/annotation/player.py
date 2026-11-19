"""Seek-friendly FFmpeg proxy, HTTP Range server, and frame-by-frame HTML5 player."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import cv2
from IPython.display import HTML, display

PHASES = [
    "start_rest", "setup", "first_pull", "second_pull", "clean",
    "middle_rest", "jerk_phase_one", "jerk_phase_two",
    "rest_up", "bring_down", "end_rest", "end_rep",
]

_vp_httpd: ThreadingHTTPServer | None = None


def new_rep_template() -> dict[str, int]:
    """Return a zero-initialized dictionary for all Clean & Jerk rep phases."""
    return {p: 0 for p in PHASES}


def tag_for(annotator_id: int) -> str:
    """Return the OpenDatasetStore processed tag for an annotator ID."""
    return f"rep_segments_a{annotator_id}"


def _free_port(start: int = 8123) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port available for video range server.")


def _make_range_handler(video_dir: str):
    class RangeHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _serve(self, head: bool = False):
            name = os.path.basename(unquote(urlparse(self.path).path))
            path = os.path.join(video_dir, name)
            if not os.path.isfile(path):
                self.send_error(404)
                return

            size = os.path.getsize(path)
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            status = 200
            if rng and rng.startswith("bytes="):
                s, _, e = rng[6:].partition("-")
                try:
                    if s == "":
                        start = max(0, size - int(e))
                    else:
                        start = int(s)
                        if e:
                            end = min(int(e), size - 1)
                    status = 206
                except ValueError:
                    pass
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return

            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "max-age=3600")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head:
                return

            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self._serve(head=False)

        def do_HEAD(self):
            self._serve(head=True)

    return RangeHandler


_PLAYER_HTML_TEMPLATE = r"""
<div id="root" tabindex="0" style="background:#18181b;color:#f4f4f5;padding:10px 12px;border-radius:10px;font-family:system-ui,sans-serif;max-width:800px;box-sizing:border-box;border:1px solid #3f3f46;outline:none;">

  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-size:13px;font-weight:bold;color:#38bdf8;">🎥 __ENTRY_ID__</span>
    <span id="readout" style="background:#27272a;padding:4px 12px;border-radius:6px;font-size:16px;font-weight:bold;color:#facc15;border:1px solid #52525b;font-variant-numeric:tabular-nums;">Frame 0 / __TOTAL_M1__</span>
  </div>

  <div style="background:#000;border-radius:6px;overflow:hidden;">
    <video id="vid" src="__VIDEO_URL__" preload="auto" muted playsinline
           style="width:100%;max-height:70vh;display:block;cursor:pointer;"></video>
  </div>

  <input type="range" id="scrub" min="0" max="__TOTAL_M1__" value="0" step="1"
         style="width:100%;height:18px;margin-top:6px;cursor:pointer;">

  <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
    <button class="b" data-d="-10">-10</button>
    <button class="b" data-d="-5">-5</button>
    <button class="b a" data-d="-1">◀ -1</button>
    <button id="play" class="b" style="background:#2563eb;font-weight:bold;min-width:84px;">▶ Play</button>
    <button class="b a" data-d="1">+1 ▶</button>
    <button class="b" data-d="5">+5</button>
    <button class="b" data-d="10">+10</button>
  </div>

  <div style="margin-top:6px;font-size:11px;color:#a1a1aa;">
    Click the player first, then: <b>Space</b> play/pause · <b>←/→</b> ±1 · <b>Shift+←/→</b> ±5 · <b>↑/↓</b> ±10
  </div>
</div>

<style>
  .b{padding:6px 12px;background:#3f3f46;color:#fff;border:none;border-radius:5px;cursor:pointer;font-size:13px;font-weight:500;touch-action:manipulation;user-select:none}
  .b:hover{filter:brightness(1.25)} .b.a{color:#facc15;font-weight:bold}
</style>

<script>
(function(){
  const FPS = __FPS__;
  let LAST  = __TOTAL_M1__;

  const root = document.getElementById("root");
  const vid  = document.getElementById("vid");
  const scrub= document.getElementById("scrub");
  const out  = document.getElementById("readout");
  const play = document.getElementById("play");

  let cur = 0, target = 0, sought = -1, seeking = false;
  const clamp = f => Math.max(0, Math.min(LAST, f|0));

  function show(f){
    cur = f;
    out.textContent = "Frame " + f + " / " + LAST;
    scrub.value = f;
  }
  function doSeek(){
    sought = target;
    seeking = true;
    vid.currentTime = (target + 0.5) / FPS;
    show(target);
  }
  function goto(f){
    target = clamp(f);
    if (!seeking) doSeek();
  }
  vid.addEventListener("seeked", function(){
    seeking = false;
    if (target !== sought) doSeek();
  });
  function step(d){
    if (!vid.paused) vid.pause();
    goto((seeking ? target : cur) + d);
  }
  function toggle(){
    if (vid.paused){
      if (cur >= LAST) goto(0);
      vid.play();
    } else vid.pause();
  }
  function onFrame(now, meta){
    if (!vid.paused) show(clamp(Math.round(meta.mediaTime * FPS)));
    vid.requestVideoFrameCallback(onFrame);
  }
  if ("requestVideoFrameCallback" in vid) vid.requestVideoFrameCallback(onFrame);
  else vid.addEventListener("timeupdate", function(){
    if (!vid.paused) show(clamp(Math.round(vid.currentTime * FPS)));
  });

  vid.addEventListener("play",  () => play.textContent = "⏸ Pause");
  vid.addEventListener("pause", () => { play.textContent = "▶ Play"; target = cur = clamp(Math.round(vid.currentTime * FPS - 0.5)); show(cur); });
  vid.addEventListener("ended", () => { play.textContent = "▶ Play"; show(LAST); });
  vid.addEventListener("loadedmetadata", function(){
    const n = Math.round(vid.duration * FPS) - 1;
    if (n > 0){ LAST = n; scrub.max = n; show(0); }
    if (window.google && google.colab && google.colab.output)
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight + 8, true);
  });

  play.onclick = toggle;
  vid.onclick  = () => { root.focus(); toggle(); };
  document.querySelectorAll("button[data-d]").forEach(b =>
    b.onclick = () => { root.focus(); step(parseInt(b.dataset.d)); });
  scrub.addEventListener("input", () => { if (!vid.paused) vid.pause(); goto(parseInt(scrub.value)); });

  root.addEventListener("keydown", function(e){
    const k = e.code;
    if (k === "Space")            { e.preventDefault(); toggle(); }
    else if (k === "ArrowRight")  { e.preventDefault(); step(e.shiftKey ? 5 : 1); }
    else if (k === "ArrowLeft")   { e.preventDefault(); step(e.shiftKey ? -5 : -1); }
    else if (k === "ArrowUp")     { e.preventDefault(); step(10); }
    else if (k === "ArrowDown")   { e.preventDefault(); step(-10); }
  });
  root.focus();
})();
</script>
"""


def render_video_player(
    store,
    entry_type: str,
    entry_id: str,
    label: str = "synced_video",
    proxy_height: int = 720,
    quality_crf: int = 22,
    force_reencode: bool = False,
    video_dir: str = "/tmp/video_player",
    port_url_fn=None,
) -> None:
    """Download video, encode seek-friendly proxy, start HTTP Range server, and display player."""
    global _vp_httpd

    os.makedirs(video_dir, exist_ok=True)
    src = store.download_related_file(
        entry_type=entry_type,
        entry_id=entry_id,
        label=label,
        dest_folder=video_dir,
    )
    if not src or not os.path.exists(src):
        raise FileNotFoundError(f"❌ No {label} for {entry_id}")

    proxy_name = f"proxy_{entry_id}_{proxy_height}p_crf{quality_crf}.mp4"
    proxy_path = os.path.join(video_dir, proxy_name)

    if (
        force_reencode
        or not os.path.exists(proxy_path)
        or os.path.getmtime(proxy_path) < os.path.getmtime(src)
    ):
        cap = cv2.VideoCapture(src)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        vf = ["-vf", f"scale=-2:{proxy_height}"] if src_h > proxy_height else []
        print(f"⚡ Encoding {proxy_height}p proxy (one-time)...")
        t0 = time.time()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src, *vf,
                "-c:v", "libx264", "-preset", "veryfast",
                "-crf", str(quality_crf),
                "-g", "6", "-bf", "0",
                "-threads", "0",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                proxy_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print(f"✅ Done in {time.time() - t0:.1f}s · {os.path.getsize(proxy_path) / 1e6:.1f} MB")
    else:
        print("♻️ Using cached proxy")

    cap = cv2.VideoCapture(proxy_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if _vp_httpd is not None:
        try:
            _vp_httpd.shutdown()
            _vp_httpd.server_close()
        except Exception:
            pass

    port = _free_port()
    _vp_httpd = ThreadingHTTPServer(("", port), _make_range_handler(video_dir))
    _vp_httpd.daemon_threads = True
    threading.Thread(target=_vp_httpd.serve_forever, daemon=True).start()

    base_url = port_url_fn(port).rstrip("/") if port_url_fn else f"http://localhost:{port}"

    video_url = f"{base_url}/{proxy_name}"
    html = (
        _PLAYER_HTML_TEMPLATE.replace("__VIDEO_URL__", video_url)
        .replace("__FPS__", repr(float(fps)))
        .replace("__TOTAL_M1__", str(total_frames - 1))
        .replace("__ENTRY_ID__", entry_id)
    )
    display(HTML(html))

