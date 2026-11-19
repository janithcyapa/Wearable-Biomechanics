"""Temporal cross-correlation synchronization between cameras and raw IMU signals."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal

from .video import MOTION_KEYPOINTS

_PUB_FONT = dict(family="Arial, Helvetica, sans-serif", size=12, color="#111827")
_PUB_COLORS = ("#0072B2", "#D55E00")


def _style_pub_fig(fig: go.Figure, title: str, x_title: str, y_title: str) -> None:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=380,
        margin=dict(l=60, r=25, t=55, b=55),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1.0, xanchor="right"),
    )
    fig.update_xaxes(
        showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
        ticks="outside", showgrid=True, gridcolor="#e5e7eb", gridwidth=0.7, zeroline=False,
    )
    fig.update_yaxes(
        showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
        ticks="outside", showgrid=True, gridcolor="#e5e7eb", gridwidth=0.7,
        zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8,
    )


def extract_movement_signal(df: pd.DataFrame, prepend_first: bool = False) -> np.ndarray:
    """Compute a 1-D movement energy signal from COCO body keypoints."""
    x_cols = [f"{name}_x" for name in MOTION_KEYPOINTS]
    y_cols = [f"{name}_y" for name in MOTION_KEYPOINTS]
    s_cols = [f"{name}_score" for name in MOTION_KEYPOINTS]

    kx = df[x_cols].copy()
    ky = df[y_cols].copy()
    ks = df[s_cols].fillna(0.0).to_numpy()

    low_conf = ks < 0.3
    kx[low_conf] = np.nan
    ky[low_conf] = np.nan

    kx = kx.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0).to_numpy()
    ky = ky.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0).to_numpy()

    if prepend_first:
        dx = np.diff(kx, axis=0, prepend=kx[:1])
        dy = np.diff(ky, axis=0, prepend=ky[:1])
        pair_conf = np.minimum(ks, np.roll(ks, 1, axis=0))
        pair_conf[0] = ks[0]
    else:
        dx = np.diff(kx, axis=0)
        dy = np.diff(ky, axis=0)
        pair_conf = np.minimum(ks[:-1], ks[1:])

    joint_speed = np.sqrt(0.25 * dx**2 + dy**2) * (pair_conf >= 0.3)
    raw_sig = np.sum(joint_speed, axis=1)
    if len(raw_sig) >= 5:
        raw_sig = signal.medfilt(raw_sig, kernel_size=5)
    return raw_sig


def calculate_sync_lag(front_df: pd.DataFrame, side_df: pd.DataFrame, plot: bool = False) -> int:
    """Calculate frame lag (`t_front - t_side`) between front and side videos."""
    sig_front = extract_movement_signal(front_df, prepend_first=False)
    sig_side = extract_movement_signal(side_df, prepend_first=False)

    sig_front_norm = (sig_front - np.mean(sig_front)) / (np.std(sig_front) + 1e-8)
    sig_side_norm = (sig_side - np.mean(sig_side)) / (np.std(sig_side) + 1e-8)

    correlation = signal.correlate(sig_front_norm, sig_side_norm, mode="full")
    lags = signal.correlation_lags(len(sig_front_norm), len(sig_side_norm), mode="full")
    best_lag = int(lags[np.argmax(correlation)])

    if plot:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=sig_front_norm, mode="lines", name="Front Movement", line=dict(width=1.4, color=_PUB_COLORS[0])))
        fig1.add_trace(go.Scatter(y=sig_side_norm, mode="lines", name="Side Movement", line=dict(width=1.4, color=_PUB_COLORS[1])))
        _style_pub_fig(fig1, "Original Un-Synced Movement Signals", "Frame Index", "Normalized Energy (Z-score)")
        fig1.show(config={"toImageButtonOptions": {"format": "png", "scale": 3}})

        fig2 = go.Figure()
        if best_lag > 0:
            x1 = np.arange(len(sig_front_norm))
            x2 = np.arange(len(sig_side_norm)) + best_lag
        else:
            x1 = np.arange(len(sig_front_norm)) - best_lag
            x2 = np.arange(len(sig_side_norm))

        fig2.add_trace(go.Scatter(x=x1, y=sig_front_norm, mode="lines", name="Front Movement (Synced)", line=dict(width=1.4, color=_PUB_COLORS[0])))
        fig2.add_trace(go.Scatter(x=x2, y=sig_side_norm, mode="lines", name="Side Movement (Synced)", line=dict(width=1.4, color=_PUB_COLORS[1])))
        _style_pub_fig(fig2, f"Aligned Movement Signals (Side Lag: {best_lag} frames)", "Aligned Frame Index", "Normalized Energy (Z-score)")
        fig2.show(config={"toImageButtonOptions": {"format": "png", "scale": 3}})

    return best_lag


def calculate_front_to_raw_sync_lag(
    front_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    front_video_path: str | None = None,
    video_fps: float | None = None,
    imu_fs: float = 100.0,
    max_lag_sec: float = 25.0,
    plot: bool = True,
) -> dict:
    """Synchronize Front Video keypoints (30Hz or 60Hz) with 100Hz Raw IMU Accelerometer data."""
    sig_front_raw = extract_movement_signal(front_df, prepend_first=True)
    sig_front_raw = np.clip(sig_front_raw, 0, np.percentile(sig_front_raw, 99.5))

    accel_cols = [c for c in raw_df.columns if c.endswith(("_AX", "_AY", "_AZ"))]
    accel_data = raw_df[accel_cols].interpolate(method="linear", limit_direction="both").fillna(0.0).to_numpy()
    n_sensors = accel_data.shape[1] // 3
    accel_3d = accel_data[:, : n_sensors * 3].reshape(-1, n_sensors, 3)

    accel_mag = np.linalg.norm(accel_3d, axis=2)
    gravity_baseline = np.median(accel_mag, axis=0, keepdims=True)
    dynamic_accel = np.abs(accel_mag - gravity_baseline)

    sig_imu_100 = np.sum(dynamic_accel, axis=1)
    if len(sig_imu_100) >= 5:
        sig_imu_100 = signal.medfilt(sig_imu_100, kernel_size=5)
    sig_imu_100 = np.clip(sig_imu_100, 0, np.percentile(sig_imu_100, 99.5))

    t_imu_100 = np.arange(len(sig_imu_100)) / float(imu_fs)
    imu_duration_s = t_imu_100[-1]

    detected_fps: float | None = None
    if video_fps is not None:
        detected_fps = float(video_fps)
    elif front_video_path and os.path.exists(front_video_path):
        cap = cv2.VideoCapture(front_video_path)
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if raw_fps and raw_fps > 0:
            detected_fps = 60.0 if raw_fps > 45.0 else 30.0

    def evaluate_at_fps(cand_fps: float):
        t_grid = np.arange(0, imu_duration_s, 1.0 / cand_fps)
        sig_imu_grid = np.interp(t_grid, t_imu_100, sig_imu_100)

        b, a = signal.butter(3, 2.0 / (0.5 * cand_fps), btype="low")
        sf_smooth = signal.filtfilt(b, a, sig_front_raw) if len(sig_front_raw) > 15 else sig_front_raw
        si_smooth = signal.filtfilt(b, a, sig_imu_grid) if len(sig_imu_grid) > 15 else sig_imu_grid

        sf_corr = (sf_smooth - np.mean(sf_smooth)) / (np.std(sf_smooth) + 1e-8)
        si_corr = (si_smooth - np.mean(si_smooth)) / (np.std(si_smooth) + 1e-8)

        sf_plot = (sig_front_raw - np.mean(sig_front_raw)) / (np.std(sig_front_raw) + 1e-8)
        si_plot = (sig_imu_grid - np.mean(sig_imu_grid)) / (np.std(sig_imu_grid) + 1e-8)

        corr = signal.correlate(sf_corr, si_corr, mode="full")
        lags = signal.correlation_lags(len(sf_corr), len(si_corr), mode="full")

        max_lag_frames = int(round(max_lag_sec * cand_fps))
        valid_mask = np.abs(lags) <= max_lag_frames
        if not np.any(valid_mask):
            valid_mask = np.ones_like(lags, dtype=bool)

        valid_lags = lags[valid_mask]
        valid_corr = corr[valid_mask]

        taper = 1.0 - 0.10 * (np.abs(valid_lags) / (max_lag_frames + 1e-8))
        best_idx = int(np.argmax(valid_corr * taper))
        best_lag_f = int(valid_lags[best_idx])

        norm_denom = np.sqrt(np.sum(sf_corr**2) * np.sum(si_corr**2)) + 1e-8
        peak_r = float(valid_corr[best_idx] / norm_denom)
        return best_lag_f, peak_r, sf_plot, si_plot

    if detected_fps is not None:
        best_lag_frames, corr_r, sig_front_norm, sig_raw_norm = evaluate_at_fps(detected_fps)
        final_fps = detected_fps
    else:
        inferred_fps_ratio = len(front_df) / max(imu_duration_s, 1.0)
        res_30 = evaluate_at_fps(30.0)
        res_60 = evaluate_at_fps(60.0)
        score_30 = res_30[1] * (1.25 if inferred_fps_ratio <= 44.0 else 0.75)
        score_60 = res_60[1] * (1.25 if inferred_fps_ratio > 44.0 else 0.75)
        if score_60 > score_30:
            final_fps = 60.0
            best_lag_frames, corr_r, sig_front_norm, sig_raw_norm = res_60
        else:
            final_fps = 30.0
            best_lag_frames, corr_r, sig_front_norm, sig_raw_norm = res_30

    best_lag_sec = best_lag_frames / final_fps
    best_lag_imu_samples = int(round(best_lag_sec * imu_fs))

    if plot:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(sig_front_norm)),
                y=sig_front_norm,
                mode="lines",
                name=f"Front Video ({int(final_fps)} Hz)",
                line=dict(width=1.4, color=_PUB_COLORS[0]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(sig_raw_norm)) + best_lag_frames,
                y=sig_raw_norm,
                mode="lines",
                name=f"Raw IMU Accel ({int(final_fps)} Hz)",
                line=dict(width=1.4, color=_PUB_COLORS[1]),
            )
        )
        _style_pub_fig(
            fig,
            f"Aligned Movement Signals — {int(final_fps)} Hz | Lag: {best_lag_frames:+d} frames ({best_lag_sec:+.3f}s) | r = {corr_r:.3f}",
            f"Front Video Frame Index ({int(final_fps)} Hz)",
            "Normalized Movement Energy (Z-score)",
        )
        fig.show(config={"toImageButtonOptions": {"format": "png", "scale": 3}})

    return {
        "video_fps": final_fps,
        "delta_t_frames": best_lag_frames,
        "delta_t_imu_samples": best_lag_imu_samples,
        "delta_t_sec": round(best_lag_sec, 4),
        "corr_peak": round(corr_r, 4),
    }

