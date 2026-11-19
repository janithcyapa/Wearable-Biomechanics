"""Publication-grade figures and tables matching Dataset Paper/Plan.md (Nature Scientific Data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .loader import load_set_h5, load_summary_tables
from .nomenclature_helpers import MARKER_ORDER

_FONT = dict(family="Arial, Helvetica, sans-serif", size=11, color="#111827")
_PALETTE = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish green
    "#CC79A7",  # Reddish purple
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
]
_PHASE_BAND_COLORS = [
    "rgba(0,114,178,0.08)", "rgba(230,159,0,0.08)", "rgba(0,158,115,0.08)",
    "rgba(213,94,0,0.08)", "rgba(204,121,167,0.08)", "rgba(86,180,233,0.08)",
    "rgba(0,114,178,0.12)", "rgba(230,159,0,0.12)", "rgba(0,158,115,0.12)",
    "rgba(213,94,0,0.12)", "rgba(204,121,167,0.12)",
]


def _pub_axes(fig: go.Figure, rows: int = 1, cols: int = 1) -> None:
    kw = dict(
        showline=True, linewidth=1.2, linecolor="#1f2937", mirror=True,
        ticks="outside", tickwidth=1, tickcolor="#1f2937",
        showgrid=True, gridcolor="#e5e7eb", gridwidth=0.6,
    )
    if rows == 1 and cols == 1:
        fig.update_xaxes(zeroline=False, **kw)
        fig.update_yaxes(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8, **kw)
        return
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(zeroline=False, row=r, col=c, **kw)
            fig.update_yaxes(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8, row=r, col=c, **kw)


def build_paper_tables(dataset_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Generate Tables 5, 7, and 8 from Plan.md directly from the exported dataset."""
    tables = load_summary_tables(dataset_dir)
    p_df = tables["participants"].copy()
    s_df = tables["sets"].copy()
    m_df = tables["phase_markers"].copy()

    # Table 5: Athlete demographics + per-participant set/lift counts
    if not s_df.empty and not p_df.empty:
        counts = s_df.groupby("participant_id", as_index=False).agg(
            sets=("set_id", "count"),
            lifts=("lifts_total", "sum"),
        )
        tab5 = p_df.merge(counts, on="participant_id", how="left")
    else:
        tab5 = p_df

    # Table 7: Release coverage by tier (Complete vs Fault-Tolerant vs Total)
    tab7_rows = []
    if not s_df.empty:
        for label, sub in [
            ("Complete (No Faults)", s_df[~s_df["has_faults"].astype(bool)]),
            ("Fault-Tolerant", s_df[s_df["has_faults"].astype(bool)]),
            ("Total", s_df),
        ]:
            tab7_rows.append(
                {
                    "tier": label,
                    "sets": int(len(sub)),
                    "participants": int(sub["participant_id"].nunique()),
                    "lifts_total": int(sub["lifts_total"].sum()),
                    "lifts_usable": int(sub["lifts_usable"].sum()),
                    "duration_min": round(float(sub["duration_s"].sum()) / 60.0, 2),
                    "total_imu_samples": int(sub["imu_samples"].sum()),
                }
            )
    tab7 = pd.DataFrame(tab7_rows)

    # Table 8: Inter-annotator agreement per phase marker
    tab8_rows = []
    if not m_df.empty:
        present = [m for m in MARKER_ORDER if m in set(m_df["marker_name"])]
        for m in present:
            sub = m_df[m_df["marker_name"] == m]
            sd_f = sub["sd_frames"].dropna()
            sd_ms = sub["sd_ms"].dropna()
            tab8_rows.append(
                {
                    "marker_name": m,
                    "n_boundaries": int(len(sub)),
                    "mean_sd_frames": round(float(sd_f.mean()), 2),
                    "mean_sd_ms": round(float(sd_ms.mean()), 1),
                    "max_sd_frames": round(float(sd_f.max()), 2),
                    "pct_within_6_frames": round(float((sd_f <= 6.0).mean() * 100.0), 1),
                    "pct_within_8_frames": round(float((sd_f <= 8.0).mean() * 100.0), 1),
                }
            )
    tab8 = pd.DataFrame(tab8_rows)

    return {"table5_demographics": tab5, "table7_coverage": tab7, "table8_agreement": tab8}


def plot_fig4_annotation_scheme(
    dataset_dir: str | Path,
    set_id: str = "set_0001",
    lift_id: int = 1,
    sensor_code: str = "US",
) -> go.Figure:
    """Plan.md Fig 4: Example IMU trace for one lift with the 11 shaded phases & 12 markers."""
    sd = load_set_h5(dataset_dir, set_id)
    s_idx = sd["sensors"].index(sensor_code)
    t = sd["time_s"]
    acc = sd["accel"][:, s_idx, :]
    acc_mag = np.linalg.norm(acc, axis=1)

    cons = sd["consensus_df"]
    lift_m = cons[cons["lift_id"] == lift_id].copy()
    order_map = {m: i for i, m in enumerate(MARKER_ORDER)}
    lift_m["_ord"] = lift_m["marker_name"].map(order_map)
    lift_m = lift_m.sort_values("_ord").reset_index(drop=True)

    t0 = float(lift_m["time_s"].min()) - 0.2
    t1 = float(lift_m["time_s"].max()) + 0.2
    mask = (t >= t0) & (t <= t1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t[mask],
            y=acc_mag[mask],
            mode="lines",
            name=f"{sensor_code} |a| (m/s²)",
            line=dict(width=1.6, color=_PALETTE[0]),
        )
    )

    # Shaded phase bands between consecutive markers
    for i in range(len(lift_m) - 1):
        x_start = float(lift_m.loc[i, "time_s"])
        x_end = float(lift_m.loc[i + 1, "time_s"])
        p_name = str(lift_m.loc[i, "marker_name"])
        fig.add_vrect(
            x0=x_start,
            x1=x_end,
            fillcolor=_PHASE_BAND_COLORS[i % len(_PHASE_BAND_COLORS)],
            layer="below",
            line_width=0,
        )
        fig.add_vline(x=x_start, line_width=1, line_dash="dot", line_color="#4b5563")
        fig.add_annotation(
            x=0.5 * (x_start + x_end),
            y=1.01,
            yref="paper",
            text=f"P{i + 1}:{p_name}",
            showarrow=False,
            textangle=-65,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=9, color="#1f2937"),
        )
    fig.add_vline(x=float(lift_m.iloc[-1]["time_s"]), line_width=1, line_dash="dot", line_color="#4b5563")

    fig.update_layout(
        title=dict(
            text=f"<b>Figure 4: 11-Phase Annotation Scheme ({set_id}, Lift {lift_id}, {sensor_code})</b>",
            x=0.5, y=0.98, font=dict(size=13),
        ),
        xaxis_title="Time (s)",
        yaxis_title="Acceleration Magnitude (m/s²)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=460,
        margin=dict(l=65, r=25, t=120, b=55),
        showlegend=False,
    )
    _pub_axes(fig)
    return fig


def plot_fig7_data_overview(dataset_dir: str | Path, ensemble_sensor: str = "US") -> go.Figure:
    """Plan.md Fig 7 (Core Figure 2): Lifts per athlete by tier, load, demographics, & phase-normalized ensemble."""
    tables = load_summary_tables(dataset_dir)
    p_df = tables["participants"].copy()
    s_df = tables["sets"].copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.17,
        horizontal_spacing=0.10,
        subplot_titles=(
            "<b>(a) Lifts per Participant (Stacked by Tier)</b>",
            "<b>(b) Set Distribution by Barbell Load</b>",
            "<b>(c) Participant Experience & Sex Split</b>",
            f"<b>(d) Time-Normalized {ensemble_sensor} |a| Ensemble</b>",
        ),
    )

    if not s_df.empty:
        s_df["tier_lbl"] = np.where(s_df["has_faults"].astype(bool), "Fault-Tolerant", "Complete")
        agg = s_df.groupby(["participant_id", "tier_lbl"], as_index=False)["lifts_total"].sum()
        pids = sorted(s_df["participant_id"].unique())
        for t_lbl, col in [("Complete", _PALETTE[2]), ("Fault-Tolerant", _PALETTE[1])]:
            sub = agg[agg["tier_lbl"] == t_lbl].set_index("participant_id").reindex(pids, fill_value=0)
            fig.add_trace(
                go.Bar(x=pids, y=sub["lifts_total"], name=t_lbl, marker_color=col),
                row=1, col=1,
            )

        load_counts = s_df["load_kg"].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f"{int(k)} kg" for k in load_counts.index],
                y=load_counts.values,
                marker_color=_PALETTE[0],
                marker_line=dict(width=0.8, color="#111827"),
                showlegend=False,
            ),
            row=1, col=2,
        )

    if not p_df.empty:
        p_df["experience_years"] = pd.to_numeric(p_df["experience_years"], errors="coerce")
        for sex_val, col in [("M", _PALETTE[0]), ("F", _PALETTE[3]), ("NA", "#9ca3af")]:
            sub = p_df[p_df["sex"].astype(str) == sex_val]["experience_years"].dropna()
            if not sub.empty:
                fig.add_trace(
                    go.Histogram(x=sub, name=f"Sex: {sex_val}", marker_color=col, opacity=0.85, nbinsx=8),
                    row=2, col=1,
                )

    # (d) Ensemble of |a| normalized 0..100% across lifts from all sets
    grid_x = np.linspace(0, 100, 101)
    curves = []
    for sid in s_df["set_id"].tolist()[:25]:  # sample up to 25 sets for fast plotting
        try:
            sd = load_set_h5(dataset_dir, sid)
            s_idx = sd["sensors"].index(ensemble_sensor)
            acc_mag = np.linalg.norm(sd["accel"][:, s_idx, :], axis=1)
            t = sd["time_s"]
            cons = sd["consensus_df"]
            for _, grp in cons.groupby("lift_id"):
                t_s = float(grp[grp["marker_name"] == "start"]["time_s"].iloc[0])
                t_e = float(grp[grp["marker_name"] == "end"]["time_s"].iloc[0])
                if t_e <= t_s:
                    continue
                m = (t >= t_s) & (t <= t_e)
                if m.sum() < 20 or np.all(np.isnan(acc_mag[m])):
                    continue
                pct = (t[m] - t_s) / (t_e - t_s) * 100.0
                interp = np.interp(grid_x, pct, pd.Series(acc_mag[m]).ffill().bfill().to_numpy())
                curves.append(interp)
        except Exception:
            continue

    if curves:
        mat = np.vstack(curves)
        mean_c = np.nanmean(mat, axis=0)
        sd_c = np.nanstd(mat, axis=0)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid_x, grid_x[::-1]]),
                y=np.concatenate([mean_c + sd_c, (mean_c - sd_c)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,114,178,0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                name="±1 SD",
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(x=grid_x, y=mean_c, mode="lines", name="Mean |a|", line=dict(width=1.8, color=_PALETTE[0])),
            row=2, col=2,
        )

    fig.update_layout(
        title=dict(text="<b>Figure 7: Clean & Jerk Dataset Overview</b>", x=0.5, font=dict(size=15)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=760,
        barmode="stack",
        margin=dict(l=60, r=25, t=75, b=70),
        legend=dict(orientation="h", y=-0.09, x=0.5, xanchor="center"),
    )
    _pub_axes(fig, rows=2, cols=2)
    fig.update_xaxes(title_text="Participant ID", tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="Lifts (count)", row=1, col=1)
    fig.update_xaxes(title_text="Barbell Load", row=1, col=2)
    fig.update_yaxes(title_text="Sets (count)", row=1, col=2)
    fig.update_xaxes(title_text="Experience (years)", row=2, col=1)
    fig.update_yaxes(title_text="Participants", row=2, col=1)
    fig.update_xaxes(title_text="Normalized Lift Progress (%)", row=2, col=2)
    fig.update_yaxes(title_text="|a| (m/s²)", row=2, col=2)
    return fig


def plot_fig8_completeness_and_plausibility(dataset_dir: str | Path) -> go.Figure:
    """Plan.md Fig 8 (Core Figure 3A): Sets x 26 Sensors valid fraction + static |a| (~9.81 m/s^2) per sensor."""
    tables = load_summary_tables(dataset_dir)
    set_ids = sorted(tables["sets"]["set_id"].tolist())

    valid_matrix = []
    sensor_names = None
    static_mags: dict[str, list[float]] = {}

    for sid in set_ids:
        sd = load_set_h5(dataset_dir, sid)
        if sensor_names is None:
            sensor_names = sd["sensors"]
            for code in sensor_names:
                static_mags[code] = []

        acc = sd["accel"]
        cons = sd["consensus_df"]
        # Use pre_rest (first 1.0s or start->setup) as static window if available
        static_end = min(100, acc.shape[0])
        row_fracs = []
        for s_idx, code in enumerate(sensor_names):
            a_sl = acc[:, s_idx, :]
            finite_mask = ~np.any(np.isnan(a_sl), axis=1)
            frac = float(finite_mask.mean()) if len(finite_mask) else 0.0
            if frac > 0 and any(np.nanstd(a_sl[:, j]) == 0 for j in range(3)):
                frac = 0.0
            row_fracs.append(frac)

            if frac > 0.5 and static_end > 10:
                mags = np.linalg.norm(a_sl[:static_end], axis=1)
                mags = mags[~np.isnan(mags)]
                if len(mags):
                    static_mags[code].append(float(np.mean(mags)))
        valid_matrix.append(row_fracs)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.14,
        subplot_titles=(
            "<b>(a) Valid Data Fraction Across All Sets × 26 IMUs</b>",
            "<b>(b) Static Accelerometer Magnitude per Sensor (Mean ± SD across Sets)</b>",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            z=valid_matrix,
            x=sensor_names,
            y=set_ids,
            colorscale=[[0.0, "#D55E00"], [0.5, "#E69F00"], [1.0, "#009E73"]],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Valid Frac", len=0.55, y=0.72),
            hovertemplate="<b>%{y} · %{x}</b><br>Valid Fraction: %{z:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    means = [float(np.mean(static_mags[c])) if static_mags[c] else np.nan for c in sensor_names]
    sds = [float(np.std(static_mags[c], ddof=1)) if len(static_mags[c]) > 1 else 0.0 for c in sensor_names]
    fig.add_trace(
        go.Scatter(
            x=sensor_names,
            y=means,
            error_y=dict(type="data", array=sds, visible=True, color="#1f2937"),
            mode="markers",
            marker=dict(size=7, color=_PALETTE[0]),
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=9.81, line_dash="dash", line_color="#009E73", annotation_text="1 g (9.81 m/s²)", row=2, col=1)

    fig.update_layout(
        title=dict(text="<b>Figure 8: Dataset Completeness & Static Signal Plausibility</b>", x=0.5, font=dict(size=14)),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=820,
        margin=dict(l=75, r=35, t=75, b=70),
    )
    _pub_axes(fig, rows=2, cols=1)
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=8), row=1, col=1)
    fig.update_xaxes(title_text="Sensor Short Code", tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text="Static |a| (m/s²)", row=2, col=1)
    return fig


def plot_fig10_annotation_agreement(dataset_dir: str | Path) -> go.Figure:
    """Plan.md Fig 10 (Core Figure 3B): Box plot of boundary SD per marker with 6-frame (200 ms) threshold."""
    m_df = load_summary_tables(dataset_dir)["phase_markers"]
    present = [m for m in MARKER_ORDER if m in set(m_df["marker_name"])]

    fig = go.Figure()
    for idx, marker in enumerate(present):
        sub = m_df[m_df["marker_name"] == marker]
        fig.add_trace(
            go.Box(
                y=sub["sd_frames"],
                name=marker,
                marker_color=_PALETTE[idx % len(_PALETTE)],
                boxmean=True,
                showlegend=False,
            )
        )

    fig.add_hline(
        y=6.0,
        line_dash="dash",
        line_color="#D55E00",
        annotation_text="6 frames (200 ms)",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text="<b>Figure 10: Inter-Annotator Boundary SD Across the 12 Phase Markers</b>", x=0.5, font=dict(size=14)),
        xaxis_title="Phase Marker",
        yaxis_title="Inter-Annotator SD (frames at 30 fps)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=460,
        margin=dict(l=65, r=25, t=60, b=95),
    )
    _pub_axes(fig)
    fig.update_xaxes(tickangle=-30)
    return fig
