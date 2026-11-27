"""Core backend for interactive post-export video anonymization / blur fixing.

Preserves exact frame count, CFR frame rate, and H.264 (libx264, yuv420p)
encoding so IMU-to-video synchronization and consensus annotations remain valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

BACKUP_DIR_NAME = "_originals_backup"
QA_STATE_FILENAME = "_blur_qa_state.json"
RECIPES_DIR_NAME = "_blur_recipes"


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    nb_frames: int
    codec: str = ""
    pix_fmt: str = ""
    has_audio: bool = False


@dataclass
class ExportResult:
    video_path: Path
    backup_path: Path
    frames_in: int
    frames_out: int
    fps_in: float
    fps_out: float
    codec_out: str
    pix_fmt_out: str


@dataclass
class BlurProject:
    """Serializable multi-layer blur recipe for a single video."""

    blur_strength: int = 55
    feather_px: int = 15
    blur_mode: str = "gaussian"  # "gaussian" | "pixelate"
    global_strokes: list[dict[str, Any]] = field(default_factory=list)
    range_strokes: list[dict[str, Any]] = field(default_factory=list)
    # keyframe_tracks: list of {"id": int, "name": str, "keyframes": {"<frame_idx>": [cx, cy, r]}}
    keyframe_tracks: list[dict[str, Any]] = field(default_factory=list)
    # frame_strokes: {"<frame_idx>": [stroke, ...]}
    frame_strokes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def has_any_edits(self) -> bool:
        if self.global_strokes or self.range_strokes:
            return True
        if any(bool(t.get("keyframes")) for t in self.keyframe_tracks):
            return True
        if any(bool(v) for v in self.frame_strokes.values()):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_strength": int(self.blur_strength),
            "feather_px": int(self.feather_px),
            "blur_mode": str(self.blur_mode),
            "global_strokes": self.global_strokes,
            "range_strokes": self.range_strokes,
            "keyframe_tracks": self.keyframe_tracks,
            "frame_strokes": self.frame_strokes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlurProject":
        return cls(
            blur_strength=int(data.get("blur_strength", 55)),
            feather_px=int(data.get("feather_px", 15)),
            blur_mode=str(data.get("blur_mode", "gaussian")),
            global_strokes=list(data.get("global_strokes", [])),
            range_strokes=list(data.get("range_strokes", [])),
            keyframe_tracks=list(data.get("keyframe_tracks", [])),
            frame_strokes=dict(data.get("frame_strokes", {})),
        )


def _parse_fps(rate_str: str) -> float:
    if not rate_str or rate_str == "0/0":
        return 30.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 30.0
    return float(rate_str)


def probe_video(video_path: str | Path) -> VideoInfo:
    """Probe exact video stream metadata using ffprobe (with OpenCV frame fallback)."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(res.stdout)
    streams = payload.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v_stream is None:
        raise RuntimeError(f"No video stream found in {path}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width = int(v_stream["width"])
    height = int(v_stream["height"])
    fps = _parse_fps(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate") or "30/1")
    codec = str(v_stream.get("codec_name", ""))
    pix_fmt = str(v_stream.get("pix_fmt", ""))

    nb_frames_raw = v_stream.get("nb_frames")
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        nb_frames = int(nb_frames_raw)
    else:
        # Count packets directly when container omits nb_frames
        cnt_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        cnt_res = subprocess.run(cnt_cmd, capture_output=True, text=True, check=False)
        txt = cnt_res.stdout.strip()
        if txt.isdigit():
            nb_frames = int(txt)
        else:
            cap = cv2.VideoCapture(str(path))
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        nb_frames=nb_frames,
        codec=codec,
        pix_fmt=pix_fmt,
        has_audio=has_audio,
    )


def _draw_stroke(mask: np.ndarray, stroke: dict[str, Any]) -> None:
    pts = stroke.get("points", [])
    if not pts:
        return
    radius = max(1, int(round(stroke.get("radius", 25))))
    color = 0 if stroke.get("erase", False) else 255
    if len(pts) == 1:
        x, y = int(round(pts[0][0])), int(round(pts[0][1]))
        cv2.circle(mask, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        return
    for i in range(len(pts) - 1):
        p0 = (int(round(pts[i][0])), int(round(pts[i][1])))
        p1 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
        cv2.line(mask, p0, p1, color, thickness=radius * 2, lineType=cv2.LINE_AA)
        cv2.circle(mask, p0, radius, color, -1, lineType=cv2.LINE_AA)
    last = (int(round(pts[-1][0])), int(round(pts[-1][1])))
    cv2.circle(mask, last, radius, color, -1, lineType=cv2.LINE_AA)


def interpolate_track_at_frame(track: dict[str, Any], frame_idx: int) -> tuple[float, float, float] | None:
    """Linearly interpolate a moving circular blur track at `frame_idx`.

    Active only between the track's first and last keyframe (inclusive).
    """
    raw_kf = track.get("keyframes", {})
    if not raw_kf:
        return None
    kf_items = sorted((int(k), v) for k, v in raw_kf.items())
    first_f, first_v = kf_items[0]
    last_f, last_v = kf_items[-1]
    if frame_idx < first_f or frame_idx > last_f:
        return None
    if frame_idx == first_f:
        return float(first_v[0]), float(first_v[1]), float(first_v[2])
    if frame_idx == last_f:
        return float(last_v[0]), float(last_v[1]), float(last_v[2])

    for i in range(len(kf_items) - 1):
        f0, v0 = kf_items[i]
        f1, v1 = kf_items[i + 1]
        if f0 <= frame_idx <= f1:
            t = (frame_idx - f0) / float(f1 - f0)
            cx = v0[0] + t * (v1[0] - v0[0])
            cy = v0[1] + t * (v1[1] - v0[1])
            r = v0[2] + t * (v1[2] - v0[2])
            return float(cx), float(cy), float(r)
    return None


def render_mask_for_frame(project: BlurProject, frame_idx: int, width: int, height: int) -> np.ndarray:
    """Composite all 4 mask layers into a single uint8 mask (0..255) for `frame_idx`."""
    mask = np.zeros((height, width), dtype=np.uint8)

    # 1. Global strokes
    for stroke in project.global_strokes:
        _draw_stroke(mask, stroke)

    # 2. Frame-range strokes
    for stroke in project.range_strokes:
        if int(stroke.get("start", 0)) <= frame_idx <= int(stroke.get("end", -1)):
            _draw_stroke(mask, stroke)

    # 3. Interpolated moving keyframe tracks
    for track in project.keyframe_tracks:
        interp = interpolate_track_at_frame(track, frame_idx)
        if interp is not None:
            cx, cy, r = interp
            cv2.circle(
                mask,
                (int(round(cx)), int(round(cy))),
                max(1, int(round(r))),
                255,
                -1,
                lineType=cv2.LINE_AA,
            )

    # 4. Single-frame overrides (brush or erase)
    for stroke in project.frame_strokes.get(str(frame_idx), []):
        _draw_stroke(mask, stroke)

    return mask


def apply_blur_to_frame(
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    blur_strength: int = 55,
    feather_px: int = 15,
    mode: str = "gaussian",
) -> np.ndarray:
    """Apply feathered Gaussian blur or pixelation only inside the bounding box of `mask_u8`."""
    if not np.any(mask_u8):
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    ys, xs = np.nonzero(mask_u8)
    k = max(3, int(blur_strength) | 1)  # ensure odd >= 3
    f_k = max(1, int(feather_px) | 1)
    pad = max(k, f_k)

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)

    roi = frame_bgr[y0:y1, x0:x1]
    roi_mask = mask_u8[y0:y1, x0:x1]

    if mode == "pixelate":
        block = max(4, k // 3)
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
        blurred_roi = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    else:
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)

    if f_k >= 3:
        soft_mask = cv2.GaussianBlur(roi_mask, (f_k, f_k), 0)
    else:
        soft_mask = roi_mask

    alpha = (soft_mask.astype(np.float32) / 255.0)[..., None]
    blended = (blurred_roi.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    out = frame_bgr.copy()
    out[y0:y1, x0:x1] = blended
    return out


def recipe_path_for_video(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.parent / RECIPES_DIR_NAME / f"{p.stem}.blur.json"


def save_recipe(video_path: str | Path, project: BlurProject) -> Path:
    rpath = recipe_path_for_video(video_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
    return rpath


def load_recipe(video_path: str | Path) -> BlurProject:
    rpath = recipe_path_for_video(video_path)
    if rpath.exists():
        return BlurProject.from_dict(json.loads(rpath.read_text(encoding="utf-8")))
    return BlurProject()


def export_blurred_video(
    video_path: str | Path,
    project: BlurProject,
    backup_dir: str | Path | None = None,
    crf: int = 17,
    preset: str = "fast",
    progress_cb: Callable[[int, int], None] | None = None,
) -> ExportResult:
    """Render blurred video via FFmpeg pipe, assert exact frame count, backup original, and replace."""
    src_path = Path(video_path)
    in_info = probe_video(src_path)

    # Always read from pristine backup if one already exists so re-edits never double-compress
    bdir = Path(backup_dir) if backup_dir is not None else src_path.parent / BACKUP_DIR_NAME
    bpath = bdir / src_path.name
    read_path = bpath if bpath.exists() else src_path
    read_info = probe_video(read_path)

    tmp_path = src_path.parent / f".{src_path.stem}.blur_tmp.mp4"
    if tmp_path.exists():
        tmp_path.unlink()

    fps_str = f"{read_info.fps:.6f}".rstrip("0").rstrip(".")
    ffmpeg_cmd = [
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
        f"{read_info.width}x{read_info.height}",
        "-r",
        fps_str,
        "-i",
        "-",
    ]
    if read_info.has_audio:
        ffmpeg_cmd += ["-i", str(read_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    ffmpeg_cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-vsync",
        "cfr",
        "-r",
        fps_str,
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    cap = cv2.VideoCapture(str(read_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for reading: {read_path}")

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames_written = 0
    try:
        assert proc.stdin is not None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = render_mask_for_frame(project, frames_written, read_info.width, read_info.height)
            out_frame = apply_blur_to_frame(
                frame,
                mask,
                blur_strength=project.blur_strength,
                feather_px=project.feather_px,
                mode=project.blur_mode,
            )
            proc.stdin.write(out_frame.tobytes())
            frames_written += 1
            if progress_cb is not None:
                progress_cb(frames_written, read_info.nb_frames)
    finally:
        cap.release()
        if proc.stdin is not None:
            proc.stdin.close()
        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        ret = proc.wait()

    if ret != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"FFmpeg encoding failed (exit {ret}): {stderr_bytes.decode('utf-8', errors='replace')}")

    out_info = probe_video(tmp_path)
    if out_info.nb_frames != read_info.nb_frames:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Frame count mismatch! Input={read_info.nb_frames}, Output={out_info.nb_frames}. "
            "Original file was NOT modified."
        )

    bdir.mkdir(parents=True, exist_ok=True)
    if not bpath.exists():
        shutil.copy2(src_path, bpath)

    os.replace(tmp_path, src_path)
    save_recipe(src_path, project)

    return ExportResult(
        video_path=src_path,
        backup_path=bpath,
        frames_in=in_info.nb_frames,
        frames_out=out_info.nb_frames,
        fps_in=in_info.fps,
        fps_out=out_info.fps,
        codec_out=out_info.codec,
        pix_fmt_out=out_info.pix_fmt,
    )


# ---------------------------------------------------------------------------
# QA State & Dataset Verification Helpers
# ---------------------------------------------------------------------------


def load_qa_state(videos_dir: str | Path) -> dict[str, dict[str, Any]]:
    state_path = Path(videos_dir) / QA_STATE_FILENAME
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def save_qa_state(videos_dir: str | Path, state: dict[str, dict[str, Any]]) -> Path:
    state_path = Path(videos_dir) / QA_STATE_FILENAME
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state_path


def set_video_qa_status(
    videos_dir: str | Path,
    filename: str,
    status: str,
    note: str = "",
) -> None:
    state = load_qa_state(videos_dir)
    state[filename] = {
        "status": status,
        "note": note,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_qa_state(videos_dir, state)


def scan_videos_folder(videos_dir: str | Path, probe_all: bool = False) -> pd.DataFrame:
    """List exported .mp4 videos in `videos_dir` with QA status and backup state."""
    vdir = Path(videos_dir)
    if not vdir.exists():
        raise FileNotFoundError(f"Videos directory not found: {vdir}")

    state = load_qa_state(vdir)
    bdir = vdir / BACKUP_DIR_NAME
    mp4s = sorted(p for p in vdir.glob("*.mp4") if not p.name.startswith("."))

    rows: list[dict[str, Any]] = []
    for idx, p in enumerate(mp4s):
        st = state.get(p.name, {})
        row: dict[str, Any] = {
            "idx": idx,
            "filename": p.name,
            "status": st.get("status", "Pending"),
            "has_backup": (bdir / p.name).exists(),
            "has_recipe": recipe_path_for_video(p).exists(),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "updated_utc": st.get("updated_utc", ""),
        }
        if probe_all:
            info = probe_video(p)
            row.update(
                {
                    "frames": info.nb_frames,
                    "fps": round(info.fps, 3),
                    "resolution": f"{info.width}x{info.height}",
                    "codec": info.codec,
                    "pix_fmt": info.pix_fmt,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def verify_all_frame_counts(videos_dir: str | Path) -> pd.DataFrame:
    """Compare every fixed/backed-up video against its original in `_originals_backup/`."""
    vdir = Path(videos_dir)
    bdir = vdir / BACKUP_DIR_NAME
    if not bdir.exists():
        return pd.DataFrame(
            columns=["filename", "orig_frames", "new_frames", "frame_match", "fps_out", "codec", "pix_fmt"]
        )

    rows = []
    for orig in sorted(bdir.glob("*.mp4")):
        cur = vdir / orig.name
        orig_info = probe_video(orig)
        if not cur.exists():
            rows.append(
                {
                    "filename": orig.name,
                    "orig_frames": orig_info.nb_frames,
                    "new_frames": -1,
                    "frame_match": False,
                    "fps_out": 0.0,
                    "codec": "MISSING",
                    "pix_fmt": "",
                }
            )
            continue
        cur_info = probe_video(cur)
        rows.append(
            {
                "filename": orig.name,
                "orig_frames": orig_info.nb_frames,
                "new_frames": cur_info.nb_frames,
                "frame_match": (orig_info.nb_frames == cur_info.nb_frames),
                "fps_out": round(cur_info.fps, 3),
                "codec": cur_info.codec,
                "pix_fmt": cur_info.pix_fmt,
            }
        )
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_checksums_sha256(dataset_dir: str | Path, videos_subdir: str = "videos") -> dict[str, int]:
    """Update SHA-256 entries for modified videos in `<dataset_dir>/checksums.sha256`."""
    droot = Path(dataset_dir)
    vdir = droot / videos_subdir
    csum_path = droot / "checksums.sha256"

    existing: dict[str, str] = {}
    order: list[str] = []
    if csum_path.exists():
        for line in csum_path.read_text(encoding="utf-8").splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            parts = line_s.split(None, 1)
            if len(parts) == 2:
                digest, rel = parts[0], parts[1].lstrip("*").strip()
                existing[rel] = digest
                order.append(rel)

    updated = 0
    added = 0
    for mp4 in sorted(p for p in vdir.glob("*.mp4") if not p.name.startswith(".")):
        rel = mp4.relative_to(droot).as_posix()
        new_digest = _sha256_file(mp4)
        if rel not in existing:
            order.append(rel)
            existing[rel] = new_digest
            added += 1
        elif existing[rel] != new_digest:
            existing[rel] = new_digest
            updated += 1

    lines = [f"{existing[rel]}  {rel}" for rel in order]
    csum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"updated": updated, "added": added, "total_entries": len(order)}
