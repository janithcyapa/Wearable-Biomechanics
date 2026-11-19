"""IMU channel extraction, health diagnostics, and publication-grade Plotly plots."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_IMU_CHANNEL_RE = re.compile(r"^.*?(?<!\d)(\d+)_IMU(\d+)_(AX|AY|AZ|GX|GY|GZ)$")

# Colorblind-safe Okabe-Ito publication palette
_PUB_PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
]

_PUB_FONT = dict(family="Arial, Helvetica, sans-serif", size=12, color="#111827")


def _apply_pub_axes(fig: go.Figure, rows: int = 1, cols: int = 1) -> None:
    """Apply clean, high-contrast research-paper axis styling to all subplots."""
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                showline=True,
                linewidth=1.2,
                linecolor="#1f2937",
                mirror=True,
                ticks="outside",
                tickwidth=1,
                tickcolor="#1f2937",
                showgrid=True,
                gridcolor="#e5e7eb",
                gridwidth=0.7,
                zeroline=False,
                row=r,
                col=c,
            )
            fig.update_yaxes(
                showline=True,
                linewidth=1.2,
                linecolor="#1f2937",
                mirror=True,
                ticks="outside",
                tickwidth=1,
                tickcolor="#1f2937",
                showgrid=True,
                gridcolor="#e5e7eb",
                gridwidth=0.7,
                zeroline=True,
                zerolinecolor="#d1d5db",
                zerolinewidth=0.8,
                row=r,
                col=c,
            )


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
) -> go.Figure | None:
    """Visualize all IMU channels as a publication-grade Plotly status matrix."""
    health_df = get_sensor_health_summary(df)
    if health_df.empty:
        print("No matching IMU channels detected.")
        return None

    status_map = {"OK": 0, "CLIPPED": 1, "NANS": 2, "FLATLINE": 3}
    health_df["Score"] = health_df["Status"].map(status_map)

    pivot = health_df.pivot(index="Device", columns="Channel", values="Score")
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.split()[1])))

    status_pivot = health_df.pivot(index="Device", columns="Channel", values="Status").reindex(pivot.index)

    # Discrete 4-bin colorscale matching Okabe-Ito status colors
    colorscale = [
        [0.00, "#009E73"], [0.25, "#009E73"],
        [0.25, "#E69F00"], [0.50, "#E69F00"],
        [0.50, "#CC79A7"], [0.75, "#CC79A7"],
        [0.75, "#D55E00"], [1.00, "#D55E00"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            customdata=status_pivot.to_numpy(),
            hovertemplate="<b>%{y} · %{x}</b><br>Status: %{customdata}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            xgap=1,
            ygap=1,
            colorbar=dict(
                title=dict(text="Status", font=dict(size=11)),
                tickvals=[0.375, 1.125, 1.875, 2.625],
                ticktext=["OK", "Clipped", "NaNs", "Flatline"],
                len=0.85,
                outlinewidth=1,
                outlinecolor="#d1d5db",
            ),
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14)),
        xaxis_title="IMU Axis Channels",
        yaxis_title="Sensor Node",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=380,
        margin=dict(l=70, r=40, t=55, b=95),
    )
    fig.update_xaxes(
        tickangle=-90,
        tickfont=dict(size=9),
        showline=True,
        linewidth=1.2,
        linecolor="#1f2937",
        mirror=True,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=10),
        showline=True,
        linewidth=1.2,
        linecolor="#1f2937",
        mirror=True,
    )
    return fig


def plot_quick_accel_check(df: pd.DataFrame, entry_id: str) -> go.Figure:
    """Plot the first 3 accelerometer channels in publication-grade styling."""
    accel_cols = [c for c in df.columns if "_AX" in c or "_AY" in c or "_AZ" in c][:3]
    fig = go.Figure()
    for idx, col in enumerate(accel_cols):
        fig.add_trace(
            go.Scatter(
                y=df[col],
                name=col,
                mode="lines",
                line=dict(width=1.4, color=_PUB_PALETTE[idx % len(_PUB_PALETTE)]),
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>Initial Accelerometer Check — {entry_id}</b>", x=0.5, font=dict(size=14)),
        xaxis_title="Sample Index",
        yaxis_title="Acceleration (m/s²)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=380,
        margin=dict(l=60, r=25, t=55, b=55),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1.0, xanchor="right"),
    )
    _apply_pub_axes(fig, rows=1, cols=1)
    return fig


def plot_imu_verification_dashboard(
    df: pd.DataFrame, time_col: str = "Timestamp"
) -> go.Figure:
    """Build a publication-grade interactive Plotly dashboard with Device switcher and Violin plots."""
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
        vertical_spacing=0.12,
        horizontal_spacing=0.07,
        subplot_titles=(
            "<b>Accelerometer Time Series</b>",
            "<b>Accel Distribution</b>",
            "<b>Gyroscope Time Series</b>",
            "<b>Gyro Distribution</b>",
        ),
    )

    t = df[time_col] if time_col in df.columns else np.arange(len(df))
    device_trace_counts: list[int] = []

    for dev in devices:
        count = 0
        acc_cols = get_columns(df, devices=dev, axes=["AX", "AY", "AZ"])
        gyro_cols = get_columns(df, devices=dev, axes=["GX", "GY", "GZ"])
        visible = dev == devices[0]

        for idx, col in enumerate(acc_cols):
            color = _PUB_PALETTE[idx % len(_PUB_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    mode="lines",
                    line=dict(width=1.3, color=color),
                    legendgroup=f"dev_{dev}",
                    visible=visible,
                ),
                row=1,
                col=1,
            )
            count += 1

        for idx, col in enumerate(acc_cols):
            color = _PUB_PALETTE[idx % len(_PUB_PALETTE)]
            fig.add_trace(
                go.Violin(
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    box_visible=True,
                    meanline_visible=True,
                    line_color=color,
                    legendgroup=f"dev_{dev}",
                    showlegend=False,
                    visible=visible,
                ),
                row=1,
                col=2,
            )
            count += 1

        for idx, col in enumerate(gyro_cols):
            color = _PUB_PALETTE[idx % len(_PUB_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    mode="lines",
                    line=dict(width=1.3, color=color),
                    legendgroup=f"dev_{dev}",
                    visible=visible,
                ),
                row=2,
                col=1,
            )
            count += 1

        for idx, col in enumerate(gyro_cols):
            color = _PUB_PALETTE[idx % len(_PUB_PALETTE)]
            fig.add_trace(
                go.Violin(
                    y=df[col],
                    name=col.split(f"{dev}_")[-1],
                    box_visible=True,
                    meanline_visible=True,
                    line_color=color,
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
                    {"title": f"<b>IMU Verification Dashboard — Device {dev}</b>"},
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=0.0,
                y=1.13,
                xanchor="left",
                yanchor="top",
                direction="down",
                showactive=True,
                bgcolor="#f9fafb",
                bordercolor="#d1d5db",
                borderwidth=1,
            )
        ],
        title=dict(
            text=f"<b>IMU Verification Dashboard — Device {devices[0]}</b>",
            x=0.5,
            y=0.98,
            font=dict(size=15),
        ),
        height=820,
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        margin=dict(l=65, r=25, t=95, b=80),
        legend=dict(
            orientation="h",
            y=-0.11,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
    )

    _apply_pub_axes(fig, rows=2, cols=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Range (m/s²)", row=1, col=2)
    fig.update_yaxes(title_text="Angular Velocity (°/s)", row=2, col=1)
    fig.update_yaxes(title_text="Range (°/s)", row=2, col=2)
    fig.update_xaxes(title_text="Timestamp / Sample", row=2, col=1)

    return fig
