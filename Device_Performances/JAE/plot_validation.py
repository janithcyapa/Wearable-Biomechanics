"""
JAE Wearable Device - Joint Angle Estimation Validation Plot
============================================================
Plots Device 7 raw acceleration data, angle estimation, and
QArm robot reference angles — all synchronized on a common
relative time axis.

Sync Strategy
-------------
- Raw + Angle data: sample index at 100 Hz  →  t_sec = (index - 1) / 100
- QArm data:        absolute wall-clock time  →  t_rel = Timestamp - Timestamp[0]
  Both signals are shifted so their first sample sits at t = 0.
  Then they are overlaid on the same time axis.  If your actual
  experiment has a known start-offset between the two streams, change
  QARM_OFFSET_SEC below (positive = qarm started later than IMU).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_FILE   = "entry_0001_none_1781856275.csv"
ANGLE_FILE = "angle_estimatino.csv"
QARM_FILE  = "qarm_imu_validation_data.csv"

# ── tuneable offset ────────────────────────────────────────────────────────────
# Set to a positive value (seconds) if the IMU recording started N seconds
# before the QArm experiment.  0 = both streams start at the same instant.
QARM_OFFSET_SEC = 0.0

# Downsample raw/angle for plotting (every Nth sample) — keeps browser snappy
DOWNSAMPLE = 10   # plot every 10th sample → effective 10 Hz display

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
raw   = pd.read_csv(RAW_FILE)
angle = pd.read_csv(ANGLE_FILE)
qarm  = pd.read_csv(QARM_FILE)

# ── build common time axes ─────────────────────────────────────────────────────
# IMU / angle: sample index (1-based) at 100 Hz
raw["t_sec"]   = (raw["Timestamp"]   - 1) / 100.0
angle["t_sec"] = (angle["Timestamp"] - 1) / 100.0

# QArm: wall-clock → relative seconds from its own start
qarm["t_rel"]  = qarm["Timestamp"] - qarm["Timestamp"].iloc[0] + QARM_OFFSET_SEC

# QArm elbow angle: radians → degrees
qarm["Elbow_deg"] = np.degrees(qarm["Elbow_rad"])

# ── downsample for performance ─────────────────────────────────────────────────
raw_ds   = raw.iloc[::DOWNSAMPLE].copy()
angle_ds = angle.iloc[::DOWNSAMPLE].copy()

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    "ax":     "#60a5fa",   # blue-400
    "ay":     "#a78bfa",   # violet-400
    "az":     "#34d399",   # emerald-400
    "hpf":    "#f97316",   # orange-500  (HPF angle estimate)
    "kl":     "#facc15",   # yellow-400  (Kalman/KL angle estimate)
    "target": "#f43f5e",   # rose-500    (QArm target)
    "elbow":  "#22d3ee",   # cyan-400    (QArm actual elbow)
    "phase":  "rgba(255,255,255,0.06)",
}

FONT = "Inter, system-ui, sans-serif"

# ── build figure (3 rows) ──────────────────────────────────────────────────────
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[
        "Device 7 — Raw Acceleration  (AX · AY · AZ)",
        "Joint Angle Estimation  (HPF  &  Kalman)",
        "QArm Reference  (Target °  &  Actual Elbow °)",
    ],
    row_heights=[0.32, 0.34, 0.34],
)

# ── row 1 · Device 7 acceleration ─────────────────────────────────────────────
for col, color, label in [
    ("device7_IMU1_AX", C["ax"], "AX (m/s²)"),
    ("device7_IMU1_AY", C["ay"], "AY (m/s²)"),
    ("device7_IMU1_AZ", C["az"], "AZ (m/s²)"),
]:
    fig.add_trace(
        go.Scattergl(
            x=raw_ds["t_sec"],
            y=raw_ds[col],
            mode="lines",
            name=label,
            line=dict(color=color, width=1.2),
            legendgroup="accel",
            legendgrouptitle_text="Acceleration" if col == "device7_IMU1_AX" else None,
            hovertemplate=f"<b>{label}</b><br>t=%{{x:.2f}} s<br>val=%{{y:.3f}}<extra></extra>",
        ),
        row=1, col=1,
    )

# ── row 2 · Angle estimation ──────────────────────────────────────────────────
for col, color, label in [
    ("knee_angle_hpf", C["hpf"], "HPF Angle (°)"),
    ("knee_angle_kl",  C["kl"],  "Kalman Angle (°)"),
]:
    fig.add_trace(
        go.Scattergl(
            x=angle_ds["t_sec"],
            y=angle_ds[col],
            mode="lines",
            name=label,
            line=dict(color=color, width=1.5),
            legendgroup="angle",
            legendgrouptitle_text="Angle Estimation" if col == "knee_angle_hpf" else None,
            hovertemplate=f"<b>{label}</b><br>t=%{{x:.2f}} s<br>angle=%{{y:.2f}}°<extra></extra>",
        ),
        row=2, col=1,
    )

# ── row 3 · QArm reference ────────────────────────────────────────────────────
fig.add_trace(
    go.Scatter(
        x=qarm["t_rel"],
        y=qarm["Target_Angle_Deg"],
        mode="lines+markers",
        name="QArm Target (°)",
        line=dict(color=C["target"], width=2, dash="dash"),
        marker=dict(size=4, color=C["target"]),
        legendgroup="qarm",
        legendgrouptitle_text="QArm Reference",
        hovertemplate="<b>Target</b><br>t=%{x:.2f} s<br>angle=%{y:.2f}°<extra></extra>",
    ),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(
        x=qarm["t_rel"],
        y=qarm["Elbow_deg"],
        mode="lines+markers",
        name="QArm Elbow (°)",
        line=dict(color=C["elbow"], width=2),
        marker=dict(size=4, color=C["elbow"]),
        legendgroup="qarm",
        hovertemplate="<b>Elbow</b><br>t=%{x:.2f} s<br>angle=%{y:.2f}°<extra></extra>",
    ),
    row=3, col=1,
)

# ── shade Test_Phase regions on row 3 ─────────────────────────────────────────
phase_colors = {
    "START":        "rgba(255, 200,  60, 0.12)",
    "SWEEP_10.0DEG":"rgba( 99, 210, 255, 0.10)",
    "SWEEP_5.0DEG": "rgba( 99, 255, 160, 0.10)",
    "SWEEP_3.0DEG": "rgba(210, 140, 255, 0.10)",
    "SWEEP_2.0DEG": "rgba(255, 130, 100, 0.10)",
    "SWEEP_1.0DEG": "rgba(255, 220, 100, 0.10)",
    "SWEEP_0.5DEG": "rgba(100, 220, 255, 0.10)",
    "SWEEP_0.25DEG":"rgba(180, 255, 130, 0.10)",
}

# Build phase segments
qarm["phase_key"] = qarm["Test_Phase"].str.extract(r'^(START_BASE|SWEEP_[\d.]+DEG)')[0]
prev_phase = None
seg_start  = qarm["t_rel"].iloc[0]
added_shapes = set()

for idx, row in qarm.iterrows():
    phase = row["phase_key"]
    if phase != prev_phase and prev_phase is not None:
        color = phase_colors.get(prev_phase, "rgba(200,200,200,0.07)")
        fig.add_vrect(
            x0=seg_start, x1=row["t_rel"],
            fillcolor=color, opacity=1,
            layer="below", line_width=0,
            row=3, col=1,
            annotation_text=prev_phase if prev_phase not in added_shapes else "",
            annotation_position="top left",
            annotation_font=dict(size=8, color="rgba(200,200,200,0.7)"),
        )
        added_shapes.add(prev_phase)
        seg_start = row["t_rel"]
    prev_phase = phase

# close last segment
if prev_phase:
    color = phase_colors.get(prev_phase, "rgba(200,200,200,0.07)")
    fig.add_vrect(
        x0=seg_start, x1=qarm["t_rel"].iloc[-1],
        fillcolor=color, opacity=1,
        layer="below", line_width=0,
        row=3, col=1,
    )

# ── axis labels ───────────────────────────────────────────────────────────────
fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1,
                 gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)")
fig.update_yaxes(title_text="Angle (°)", row=2, col=1,
                 gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)")
fig.update_yaxes(title_text="Angle (°)", row=3, col=1,
                 gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)")
fig.update_xaxes(title_text="Relative Time (s)", row=3, col=1,
                 gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.15)")
for r in [1, 2]:
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=r, col=1)

# ── layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text="<b>Wearable Device Validation — Joint Angle Estimation vs QArm Reference</b>"
             "<br><sup>Device 7 · IMU Acceleration · HPF & Kalman Estimates · QArm Ground Truth</sup>",
        x=0.5, xanchor="center",
        font=dict(size=18, color="#f1f5f9", family=FONT),
    ),
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#cbd5e1", family=FONT, size=12),
    legend=dict(
        bgcolor="rgba(15,23,42,0.85)",
        bordercolor="rgba(148,163,184,0.25)",
        borderwidth=1,
        font=dict(size=11),
        groupclick="toggleitem",
        tracegroupgap=8,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="#334155",
        font=dict(family=FONT, size=12, color="#f1f5f9"),
    ),
    height=900,
    margin=dict(l=70, r=30, t=110, b=60),
    # Range slider on bottom x axis
    xaxis3=dict(
        rangeslider=dict(
            visible=True,
            bgcolor="#1e293b",
            bordercolor="#334155",
            thickness=0.04,
        ),
    ),
)

# Style subplot title text
for ann in fig.layout.annotations:
    ann.font = dict(size=13, color="#94a3b8", family=FONT)

# ── save & open ───────────────────────────────────────────────────────────────
out_html = "validation_plot.html"
fig.write_html(
    out_html,
    include_plotlyjs="cdn",
    config={
        "displayModeBar": True,
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "svg",
            "filename": "jae_validation",
            "height": 900,
            "width": 1600,
            "scale": 2,
        },
    },
)
print(f"✅  Plot saved → {out_html}")

try:
    import webbrowser, os
    webbrowser.open("file://" + os.path.abspath(out_html))
    print("🌐  Opened in browser")
except Exception:
    print("Open validation_plot.html manually in your browser.")
