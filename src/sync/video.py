"""Video resampling, YOLO keypoint extraction, and FFmpeg video generation."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading

import cv2
import numpy as np
import pandas as pd

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MOTION_KEYPOINTS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_INDICES = np.array(
    [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
        (13, 15), (12, 14), (14, 16),
    ],
    dtype=np.int32,
)

_yolo_lock = threading.Lock()
_yolo_model = None
_yolo_weights: str | None = None


def _get_yolo_model(weights: str = "yolo11x-pose.pt"):
    """Lazily load and cache the Ultralytics YOLO pose model."""
    global _yolo_model, _yolo_weights
    if _yolo_model is None or _yolo_weights != weights:
        from ultralytics import YOLO

        _yolo_model = YOLO(weights)
        _yolo_weights = weights
    return _yolo_model


def _runtime_device_config():
    import torch

    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    use_half = bool(has_cuda)
    default_batch = 32 if has_cuda else 8
    return has_cuda, device, use_half, default_batch


def ensure_30fps_video(video_path: str | None, target_fps: float = 30.0) -> tuple[str | None, bool]:
    """Resample >45fps videos down to `target_fps` in-place via FFmpeg."""
    if not video_path or not os.path.exists(video_path):
        return None, False

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if fps <= 45.0:
        return video_path, False

    print(
        f"⚠ Detected {fps:.2f} fps in {os.path.basename(video_path)}. "
        f"Resampling down to {target_fps:.0f} fps..."
    )
    temp_path = video_path.replace(".mp4", "_30fps_tmp.mp4")
    cmd = (
        f'ffmpeg -y -threads 4 -i "{video_path}" '
        f'-vf "fps=fps={target_fps}" '
        f'-c:v libx264 -preset ultrafast -crf 18 -an "{temp_path}"'
    )
    res = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode == 0 and os.path.exists(temp_path):
        shutil.move(temp_path, video_path)
        print(f"✓ Resampled to {target_fps:.0f} fps: {video_path}")
        return video_path, True

    print(f"⚠ FFmpeg resample failed for {video_path}, keeping original.")
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return video_path, False


def _select_foreground_person(results, kpts_all: np.ndarray, confs: np.ndarray, min_conf: float) -> int:
    num_people = len(kpts_all)
    if num_people == 1:
        return 0

    if results.boxes is not None and len(results.boxes.xywh) == num_people:
        xywh = results.boxes.xywh.cpu().numpy()
        areas = xywh[:, 2] * xywh[:, 3]
        box_confs = results.boxes.conf.cpu().numpy()
    else:
        areas = np.zeros(num_people, dtype=np.float32)
        box_confs = np.max(confs, axis=1)
        for p_idx in range(num_people):
            valid_pts = kpts_all[p_idx][confs[p_idx] >= min_conf]
            valid_pts = valid_pts[(valid_pts[:, 0] > 0) & (valid_pts[:, 1] > 0)]
            if len(valid_pts) >= 2:
                w = np.max(valid_pts[:, 0]) - np.min(valid_pts[:, 0])
                h = np.max(valid_pts[:, 1]) - np.min(valid_pts[:, 1])
                areas[p_idx] = max(w, 10.0) * max(h, 10.0)

    top6_kp_conf = np.mean(np.sort(confs, axis=1)[:, -6:], axis=1)
    valid_mask = (box_confs >= 0.30) & (top6_kp_conf >= min_conf)
    if np.any(valid_mask):
        return int(np.argmax(np.where(valid_mask, areas, -1.0)))
    return int(np.argmax(areas))


def analyze_keypoints(
    video_path: str,
    batch_size: int | None = None,
    min_conf: float = 0.25,
    weights: str = "yolo11x-pose.pt",
) -> pd.DataFrame:
    """Thread-safe batched YOLO keypoint extraction selecting the largest foreground person."""
    has_cuda, device, use_half, default_batch = _runtime_device_config()
    if batch_size is None:
        batch_size = default_batch
    elif not has_cuda and batch_size > 8:
        batch_size = 8

    all_frames_data: list[dict] = []

    with _yolo_lock:
        model = _get_yolo_model(weights)
        results_generator = model.predict(
            source=video_path,
            batch=batch_size,
            stream=True,
            verbose=False,
            device=device,
            half=use_half,
        )

        for frame_idx, results in enumerate(results_generator):
            frame_data: dict[str, float | int] = {"frame": frame_idx}

            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                kpts_all = results.keypoints.xy.cpu().numpy()
                confs = results.keypoints.conf.cpu().numpy()
                best_idx = _select_foreground_person(results, kpts_all, confs, min_conf)
                kpts = kpts_all[best_idx]
                scores = confs[best_idx]

                if len(kpts) == 17:
                    for i, name in enumerate(KEYPOINT_NAMES):
                        x, y, s = float(kpts[i][0]), float(kpts[i][1]), float(scores[i])
                        if s < min_conf or (x == 0.0 and y == 0.0):
                            frame_data[f"{name}_x"] = np.nan
                            frame_data[f"{name}_y"] = np.nan
                            frame_data[f"{name}_score"] = s
                        else:
                            frame_data[f"{name}_x"] = x
                            frame_data[f"{name}_y"] = y
                            frame_data[f"{name}_score"] = s
                    all_frames_data.append(frame_data)
                    continue

            for name in KEYPOINT_NAMES:
                frame_data[f"{name}_x"] = np.nan
                frame_data[f"{name}_y"] = np.nan
                frame_data[f"{name}_score"] = 0.0
            all_frames_data.append(frame_data)

    df = pd.DataFrame(all_frames_data)
    output_json_path = video_path.replace(".mp4", "_keypoints.json")
    df.to_json(output_json_path, orient="records")
    print(f"Keypoints saved to {output_json_path}")
    return df


def generate_annotated_video(
    video_path: str, keypoints_df: pd.DataFrame, confidence_threshold: float = 0.4
) -> str:
    """Draw skeleton overlays onto `video_path` using `keypoints_df`."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = video_path.replace(".mp4", "_annotated.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(keypoints_df))
    all_kpts_x = keypoints_df[[f"{n}_x" for n in KEYPOINT_NAMES]].to_numpy()
    all_kpts_y = keypoints_df[[f"{n}_y" for n in KEYPOINT_NAMES]].to_numpy()
    all_scores = keypoints_df[[f"{n}_score" for n in KEYPOINT_NAMES]].to_numpy()

    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        scores = all_scores[frame_idx]
        kx = all_kpts_x[frame_idx]
        ky = all_kpts_y[frame_idx]
        valid = (scores >= confidence_threshold) & (~np.isnan(kx)) & (~np.isnan(ky))

        for p1, p2 in SKELETON_INDICES:
            if valid[p1] and valid[p2]:
                cv2.line(
                    frame,
                    (int(kx[p1]), int(ky[p1])),
                    (int(kx[p2]), int(ky[p2])),
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_4,
                )

        for i in np.where(valid)[0]:
            cv2.circle(frame, (int(kx[i]), int(ky[i])), 4, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to {out_path}")
    return out_path


def generate_synced_video(
    front_video_path: str | None,
    side_video_path: str | None,
    side_lag_frames: int | None = None,
) -> str | None:
    """Create a frame-accurate side-by-side video (or half-black when one camera is missing)."""
    has_front = bool(front_video_path and os.path.exists(front_video_path))
    has_side = bool(side_video_path and os.path.exists(side_video_path))
    if not has_front and not has_side:
        print("Error: No valid video paths provided to generate_synced_video.")
        return None

    base_dir = os.path.dirname(front_video_path if has_front else side_video_path)
    out_path = os.path.join(base_dir, "synced_side_by_side.mp4")
    if os.path.exists(out_path):
        os.remove(out_path)

    ref_path = front_video_path if has_front else side_video_path
    cap = cv2.VideoCapture(ref_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if has_front and has_side:
        lag = int(side_lag_frames) if side_lag_frames is not None else 0
        if lag > 0:
            trim_f = f"trim=start_frame={lag},setpts=PTS-STARTPTS,"
            trim_s = "setpts=PTS-STARTPTS,"
        elif lag < 0:
            trim_f = "setpts=PTS-STARTPTS,"
            trim_s = f"trim=start_frame={abs(lag)},setpts=PTS-STARTPTS,"
        else:
            trim_f = "setpts=PTS-STARTPTS,"
            trim_s = "setpts=PTS-STARTPTS,"

        filter_complex = (
            f"[0:v]{trim_f}fps={fps:.4f},scale=-2:720[v0];"
            f"[1:v]{trim_s}fps={fps:.4f},scale=-2:720[v1];"
            f"[v0][v1]hstack=inputs=2:shortest=1[v]"
        )
        cmd = (
            f'ffmpeg -y -threads 4 -i "{front_video_path}" -i "{side_video_path}" '
            f'-filter_complex "{filter_complex}" '
            f'-map "[v]" -c:v libx264 -preset ultrafast -crf 23 -an "{out_path}"'
        )
    elif has_front:
        filter_complex = (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps:.4f},scale=-2:720,pad=iw*2:ih:0:0:black[v]"
        )
        cmd = (
            f'ffmpeg -y -threads 4 -i "{front_video_path}" '
            f'-filter_complex "{filter_complex}" '
            f'-map "[v]" -c:v libx264 -preset ultrafast -crf 23 -an "{out_path}"'
        )
    else:
        filter_complex = (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps:.4f},scale=-2:720,pad=iw*2:ih:iw:0:black[v]"
        )
        cmd = (
            f'ffmpeg -y -threads 4 -i "{side_video_path}" '
            f'-filter_complex "{filter_complex}" '
            f'-map "[v]" -c:v libx264 -preset ultrafast -crf 23 -an "{out_path}"'
        )

    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0 and os.path.exists(out_path):
        print(f"Synced video saved to {out_path}")
        return out_path

    print("FFmpeg synchronization fallback: check video input paths.")
    return None

