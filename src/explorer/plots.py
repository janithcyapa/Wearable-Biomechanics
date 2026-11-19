"""Publication-grade Plotly visualizations for Clean&JerkDataset summaries and Set .h5 files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PUB_FONT = dict(family="Arial, Helvetica, sans-serif", size=12, color="#111827")
_PUB_PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
]


def _style_pub_axes(fig: go.Figure, rows: int = 1, cols: int = 1) -> None:
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
                ticks="outside", showgrid=True, gridcolor="#e5e7eb", gridwidth=0.7,
                zeroline=False, row=r, col=c,
            )
            fig.update_yaxes(
                showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
                ticks="outside", showgrid=True, gridcolor="#e5e7eb", gridwidth=0.7,
                zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8, row=r, col=c,
            )


def plot_cohort_overview(participants_df: pd.DataFrame, sets_df: pd.DataFrame) -> go.Figure:
    """Plot cohort anthropometrics, load distribution, and sets per participant."""
    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
        subplot_titles=(
            "<b>Body Mass vs. Stature</b>",
            "<b>Experience Distribution (Years)</b>",
            "<b>Barbell Load Distribution (kg)</b>",
            "<b>Sets & Lifts per Participant</b>",
        ),
    )

    if not participants_df.empty:
        p = participants_df.copy()
        for col in ("height_m", "body_mass_kg", "experience_years", "age_years"):
            if col in p.columns:
                p[col] = pd.to_numeric(p[col], errors="coerce")

        valid_hw = p.dropna(subset=["height_m", "body_mass_kg"])
        fig.add_trace(
            go.Scatter(
                x=valid_hw["height_m"],
                y=valid_hw["body_mass_kg"],
                text=valid_hw["participant_id"],
                mode="markers+text",
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(size=9, color=_PUB_PALETTE[0], line=dict(width=1, color="#111827")),
                name="Participants",
                showlegend=False,
            ),
            row=1, col=1,
        )

        valid_exp = p["experience_years"].dropna()
        fig.add_trace(
            go.Histogram(
                x=valid_exp,
                nbinsx=10,
                marker_color=_PUB_PALETTE[2],
                marker_line=dict(width=1, color="#111827"),
                name="Experience",
                showlegend=False,
            ),
            row=1, col=2,
        )

    if not sets_df.empty:
        loads = pd.to_numeric(sets_df["load_kg"], errors="coerce").dropna()
        fig.add_trace(
            go.Histogram(
                x=loads,
                nbinsx=12,
                marker_color=_PUB_PALETTE[1],
                marker_line=dict(width=1, color="#111827"),
                name="Load (kg)",
                showlegend=False,
            ),
            row=2, col=1,
        )

        per_p = (
            sets_df.groupby("participant_id", as_index=False)
            .agg(sets=("set_id", "count"), lifts=("lifts_total", "sum"))
            .sort_values("participant_id")
        )
        fig.add_trace(
            go.Bar(
                x=per_p["participant_id"],
                y=per_p["sets"],
                name="Sets",
                marker_color=_PUB_PALETTE[0],
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Bar(
                x=per_p["participant_id"],
                y=per_p["lifts"],
                name="Total Lifts",
                marker_color=_PUB_PALETTE[4],
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title=dict(text="<b>Clean & Jerk Dataset — Cohort & Recording Overview</b>", x=0.5, font=dict(size=15)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=720,
        barmode="group",
        margin=dict(l=60, r=25, t=75, b=65),
        legend=dict(orientation="h", y=-0.09, x=0.5, xanchor="center"),
    )
    _style_pub_axes(fig, rows=2, cols=2)
    fig.update_xaxes(title_text="Height (m)", row=1, col=1)
    fig.update_yaxes(title_text="Body Mass (kg)", row=1, col=1)
    fig.update_xaxes(title_text="Experience (years)", row=1, col=2)
    fig.update_yaxes(title_text="Participants", row=1, col=2)
    fig.update_xaxes(title_text="Barbell Load (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Number of Sets", row=2, col=1)
    fig.update_xaxes(title_text="Participant ID", tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    return fig


def plot_sensor_health_overview(sets_df: pd.DataFrame) -> go.Figure:
    """Plot healthy sensor counts per Set across the dataset."""
    s = sets_df.sort_values("set_id").copy()
    colors = [
        _PUB_PALETTE[1] if bool(hf) else (_PUB_PALETTE[2] if ok == 26 else _PUB_PALETTE[4])
        for hf, ok in zip(s["has_faults"], s["sensors_ok"])
    ]
    fig = go.Figure(
        data=go.Bar(
            x=s["set_id"],
            y=s["sensors_ok"],
            marker_color=colors,
            marker_line=dict(width=0.5, color="#111827"),
            hovertemplate="<b>%{x}</b><br>Healthy IMUs: %{y}/26<extra></extra>",
        )
    )
    fig.add_hline(y=26, line_dash="dash", line_color="#374151", annotation_text="26/26 IMUs")
    fig.update_layout(
        title=dict(
            text="<b>Sensor Completeness per Set (Green = 26/26 Complete, Orange = Partial, Vermilion = Fault-Tolerant)</b>",
            x=0.5,
            font=dict(size=13),
        ),
        xaxis_title="Set ID",
        yaxis_title="Healthy IMUs (out of 26)",
        yaxis_range=[0, 28],
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=400,
        margin=dict(l=60, r=25, t=60, b=90),
    )
    _style_pub_axes(fig, rows=1, cols=1)
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    return fig


def plot_marker_agreement_summary(phase_markers_df: pd.DataFrame) -> go.Figure:
    """Plot inter-annotator SD (ms) across all 12 standardized phase markers."""
    if phase_markers_df.empty:
        return go.Figure()

    marker_order = [
        "start", "setup", "first_pull", "second_pull", "clean",
        "intermediate_rest", "jerk_split", "jerk_recovery",
        "overhead_hold", "descend", "post_rest", "end",
    ]
    present_order = [m for m in marker_order if m in set(phase_markers_df["marker_name"])]

    fig = go.Figure()
    for idx, marker in enumerate(present_order):
        sub = phase_markers_df[phase_markers_df["marker_name"] == marker]
        fig.add_trace(
            go.Box(
                y=sub["sd_ms"],
                name=marker,
                marker_color=_PUB_PALETTE[idx % len(_PUB_PALETTE)],
                boxmean=True,
                showlegend=False,
            )
        )

    fig.add_hline(
        y=266.7,
        line_dash="dash",
        line_color="#D55E00",
        annotation_text="8 frames (266.7 ms)",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text="<b>Inter-Annotator Agreement Across the 12 Phase Markers</b>", x=0.5, font=dict(size=14)),
        xaxis_title="Phase Marker",
        yaxis_title="Inter-Annotator SD (ms)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=440,
        margin=dict(l=60, r=25, t=60, b=95),
    )
    _style_pub_axes(fig, rows=1, cols=1)
    fig.update_xaxes(tickangle=-30)
    return fig


def plot_set_sensor_status(set_data: dict) -> go.Figure:
    """Visualize which of the 26 IMUs in a Set are healthy, flatlined, or missing."""
    sensors = set_data["sensors"]
    acc = set_data["accel"]
    gyr = set_data["gyro"]
    statuses = []
    scores = []
    for s_idx, _code in enumerate(sensors):
        a_sl, g_sl = acc[:, s_idx, :], gyr[:, s_idx, :]
        if np.all(np.isnan(a_sl)) and np.all(np.isnan(g_sl)):
            statuses.append("Missing / All-NaN")
            scores.append(2)
        elif np.any(np.isnan(a_sl)) or np.any(np.isnan(g_sl)) or any(
            np.nanstd(a_sl[:, j]) == 0 for j in range(3)
        ) or any(np.nanstd(g_sl[:, j]) == 0 for j in range(3)):
            statuses.append("Flatline / Partial NaN")
            scores.append(1)
        else:
            statuses.append("OK")
            scores.append(0)

    colorscale = [
        [0.0, "#009E73"], [0.33, "#009E73"],
        [0.33, "#E69F00"], [0.66, "#E69F00"],
        [0.66, "#D55E00"], [1.0, "#D55E00"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=[scores],
            x=sensors,
            y=[set_data["set_id"]],
            customdata=[statuses],
            hovertemplate="<b>%{x}</b><br>Status: %{customdata}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            xgap=2,
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>26-IMU Sensor Status — {set_data['set_id']}</b>", x=0.5, font=dict(size=13)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=170,
        margin=dict(l=75, r=20, t=50, b=45),
    )
    return fig


def plot_set_imu_with_markers(
    set_data: dict,
    sensor_code: str = "RKP",
    lift_id: int | None = None,
) -> go.Figure:
    """Plot Accel (m/s^2) and Gyro (deg/s) for `sensor_code` with consensus phase markers overlaid."""
    sensors = set_data["sensors"]
    if sensor_code not in sensors:
        raise ValueError(f"Sensor '{sensor_code}' not in {sensors}")

    s_idx = sensors.index(sensor_code)
    t = set_data["time_s"]
    acc = set_data["accel"][:, s_idx, :]
    gyr = set_data["gyro"][:, s_idx, :]
    cons_df = set_data["consensus_df"]

    if lift_id is not None and not cons_df.empty:
        cons_plot = cons_df[cons_df["lift_id"] == lift_id]
        if not cons_plot.empty:
            t_min = float(cons_plot["time_s"].min()) - 0.5
            t_max = float(cons_plot["time_s"].max()) + 0.5
            mask = (t >= t_min) & (t <= t_max)
            t, acc, gyr = t[mask], acc[mask], gyr[mask]
    else:
        cons_plot = cons_df

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"<b>{sensor_code} Accelerometer (m/s²)</b>",
            f"<b>{sensor_code} Gyroscope (deg/s)</b>",
        ),
    )

    for ax_i, ax_name in enumerate(("X", "Y", "Z")):
        fig.add_trace(
            go.Scatter(
                x=t, y=acc[:, ax_i], mode="lines", name=f"A{ax_name}",
                line=dict(width=1.3, color=_PUB_PALETTE[ax_i]),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=gyr[:, ax_i], mode="lines", name=f"G{ax_name}",
                line=dict(width=1.3, color=_PUB_PALETTE[ax_i + 3]),
            ),
            row=2, col=1,
        )

    if not cons_plot.empty:
        for _, row in cons_plot.iterrows():
            ts = float(row["time_s"])
            label = f"L{int(row['lift_id'])}:{row['marker_name']}"
            fig.add_vline(
                x=ts,
                line_width=1,
                line_dash="dot",
                line_color="#4b5563",
                annotation_text=label if lift_id is not None else "",
                annotation_textangle=-90,
                annotation_font_size=9,
            )

    lift_title = f"Lift {lift_id}" if lift_id is not None else "All Lifts"
    fig.update_layout(
        title=dict(text=f"<b>{set_data['set_id']} — {sensor_code} ({lift_title})</b>", x=0.5, font=dict(size=14)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_PUB_FONT,
        height=620,
        hovermode="x unified",
        margin=dict(l=65, r=25, t=70, b=60),
        legend=dict(orientation="h", y=-0.10, x=0.5, xanchor="center"),
    )
    _style_pub_axes(fig, rows=2, cols=1)
    fig.update_yaxes(title_text="Accel (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Gyro (deg/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    return fig
