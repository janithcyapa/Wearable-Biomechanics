"""IMU channel extraction, health diagnostics, and interactive Plotly dashboards."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

_IMU_CHANNEL_RE = re.compile(r"^.*?(?<!\d)(\d+)_IMU(\d+)_(AX|AY|AZ|GX|GY|GZ)$")


def get_columns(df: pd.DataFrame, devices=None, imus=None, axes=None) -> list[str]:
    """Extract sensor columns matching device IDs, IMU indices, and axes."""
    if devices is None:
        devices = [r"\d+"]
    elif not isinstance(devices, list):
        devices = [devices]

    if imus is None:
        imus = [r"\d+"]
    elif not isinstance(imus, list):
        imus = [imus]

    if axes is None:
        axes = [r"[A-Z0-9]+"]
    elif not isinstance(axes, list):
        axes = [axes]

    valid_columns: list[str] = []
    for dev in devices:
        for imu in imus:
            for axis in axes:
                pattern = rf"^.*?(?<!\d){dev}_IMU{imu}_{axis}$"
                for col in df.columns:
                    if re.match(pattern, col) and col not in valid_columns:
                        valid_columns.append(col)
    return valid_columns


def get_sensor_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Scan IMU channels for flatlines, NaNs, and extreme clipping."""
    records = []
    for col in df.columns:
        m = _IMU_CHANNEL_RE.match(col)
        if not m:
            continue
        dev, imu, axis = int(m.group(1)), int(m.group(2)), m.group(3)
        series = df[col]

        s_min, s_max, s_std = series.min(), series.max(), series.std()
        nan_count = int(series.isna().sum())
        is_flat = bool(s_std == 0 or series.nunique() <= 1)

        is_acc = axis.startswith("A")
        clipped = bool(
            (s_min < -80 or s_max > 80) if is_acc else (s_min < -2000 or s_max > 2000)
        )

        if is_flat:
            status = "FLATLINE"
        elif nan_count > 0:
            status = "NANS"
        elif clipped:
            status = "CLIPPED"
        else:
            status = "OK"

        records.append(
            {
                "Device": f"Dev {dev}",
                "Channel": f"IMU{imu}_{axis}",
                "Status": status,
                "Std": s_std,
                "Min": s_min,
                "Max": s_max,
                "Column": col,
            }
        )

    return pd.DataFrame(records)


def plot_health_matrix(
    df: pd.DataFrame, title: str = "Sensor Channel Quality Matrix"
) -> None:
    """Visualize all IMU channels as a status heatmap (OK / Clipped / NaNs / Flatline)."""
    health_df = get_sensor_health_summary(df)
    if health_df.empty:
        print("No matching IMU channels detected.")
        return

    status_map = {"OK": 0, "CLIPPED": 1, "NANS": 2, "FLATLINE": 3}
    health_df["Score"] = health_df["Status"].map(status_map)

    pivot = health_df.pivot(index="Device", columns="Channel", values="Score")
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.split()[1])))

    plt.figure(figsize=(18, 5))
    cmap = sns.color_palette(["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"])
    sns.heatmap(pivot, cmap=cmap, cbar=False, linewidths=0.5, linecolor="#eeeeee")

    plt.title(
        f"{title} (Green = OK, Orange = Clipped, Red = Flatlined/Dead)", fontsize=13
    )
    plt.xlabel("IMU Axis Channels")
    plt.ylabel("Devices")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_quick_accel_check(df: pd.DataFrame, entry_id: str) -> go.Figure:
    """Plot the first 3 accelerometer channels for a quick post-upload sanity check."""
    accel_cols = [c for c in df.columns if "_AX" in c or "_AY" in c or "_AZ" in c][:3]
    fig = go.Figure()
    for col in accel_cols:
        fig.add_trace(go.Scatter(y=df[col], name=col, mode="lines"))

    fig.update_layout(
        title=f"Initial Data Check: {entry_id}",
        xaxis_title="Samples",
        yaxis_title="Acceleration (m/s²)",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_imu_verification_dashboard(
    df: pd.DataFrame, time_col: str = "Timestamp"
) -> go.Figure:
    """Build an interactive Plotly dashboard with a Device dropdown and Violin plots."""
    devices = sorted(
        {int(_IMU_CHANNEL_RE.match(c).group(1)) for c in df.columns if _IMU_CHANNEL_RE.match(c)}
    )
    if not devices:
        raise ValueError("No matching device columns found in dataframe.")

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.72, 0.28],
        shared_xaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.06,
        subplot_titles=(
            "Accelerometer Signals (AX, AY, AZ)",
            "Accelerometer Distribution",
            "Gyroscope Signals (GX, GY, GZ)",
            "Gyroscope Distribution",
        ),
    )

    t = df[time_col] if time_col in df.columns else np.arange(len(df))
    device_trace_counts: list[int] = []

    for dev in devices:
        count = 0
        acc_cols = get_columns(df, devices=dev, axes=["AX", "AY", "AZ"])
        gyro_cols = get_columns(df, devices=dev, axes=["GX", "GY", "GZ"])
        visible = dev == devices[0]

        for col in acc_cols:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    mode="lines",
                    line=dict(width=1.2),
                    legendgroup=f"dev_{dev}",
                    visible=visible,
                ),
                row=1,
                col=1,
            )
            count += 1

        for col in acc_cols:
            fig.add_trace(
                go.Violin(
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    box_visible=True,
                    meanline_visible=True,
                    legendgroup=f"dev_{dev}",
                    showlegend=False,
                    visible=visible,
                ),
                row=1,
                col=2,
            )
            count += 1

        for col in gyro_cols:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    mode="lines",
                    line=dict(width=1.2),
                    legendgroup=f"dev_{dev}",
                    visible=visible,
                ),
                row=2,
                col=1,
            )
            count += 1

        for col in gyro_cols:
            fig.add_trace(
                go.Violin(
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    box_visible=True,
                    meanline_visible=True,
                    legendgroup=f"dev_{dev}",
                    showlegend=False,
                    visible=visible,
                ),
                row=2,
                col=2,
            )
            count += 1

        device_trace_counts.append(count)

    buttons = []
    cum_traces = [0] + list(np.cumsum(device_trace_counts))
    for i, dev in enumerate(devices):
        visibility = [False] * cum_traces[-1]
        for idx in range(cum_traces[i], cum_traces[i + 1]):
            visibility[idx] = True

        buttons.append(
            dict(
                label=f"Device {dev}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"IMU Verification Dashboard — Device {dev}"},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                direction="down",
                showactive=True,
            )
        ],
        title=dict(text=f"IMU Verification Dashboard — Device {devices[0]}", x=0.5, y=0.98),
        height=850,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.12, x=0.0),
    )

    fig.update_yaxes(title_text="Accel (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Value Range", row=1, col=2)
    fig.update_yaxes(title_text="Gyro (°/s)", row=2, col=1)
    fig.update_yaxes(title_text="Value Range", row=2, col=2)
    fig.update_xaxes(title_text="Timestamp / Sample", row=2, col=1)

    return fig

