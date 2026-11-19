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

    if not s_df.empty and not p_df.empty:
        counts = s_df.groupby("participant_id", as_index=False).agg(
            sets=("set_id", "count"),
            lifts=("lifts_total", "sum"),
        )
        tab5 = p_df.merge(counts, on="participant_id", how="left")
    else:
        tab5 = p_df

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


def _parse_lift_ids(lift_spec: int | str | list[int] | tuple[int, ...], available_lifts: list[int]) -> list[int]:
    """Parse lift specification like 1, '1,2', '1-5', or 'all' into sorted lift IDs."""
    if isinstance(lift_spec, int):
        return [lift_spec]
    if isinstance(lift_spec, (list, tuple)):
        return sorted({int(x) for x in lift_spec})
    s = str(lift_spec).strip().lower()
    if s in {"", "all", "0"}:
        return sorted(available_lifts)
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a.strip()), int(b.strip()) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def plot_fig4_annotation_scheme(
    dataset_dir: str | Path,
    set_id: str = "set_0003",
    lift_ids: int | str | list[int] | tuple[int, ...] = "1",
    sensor_codes: str | list[str] | tuple[str, ...] = "US",
    modality: str = "gyro",
    signal_mode: str = "axes",
    axes: tuple[str, ...] | list[str] | str = ("X", "Y", "Z"),
) -> go.Figure:
    """Plan.md Fig 4: 11 shaded phases & 12 markers over 1+ sensors, 1+ lifts, Accel/Gyro/Both."""
    if modality not in {"accel", "gyro", "both"}:
        raise ValueError("modality must be 'accel', 'gyro', or 'both'")
    if signal_mode not in {"magnitude", "axes", "both"}:
        raise ValueError("signal_mode must be 'magnitude', 'axes', or 'both'")

    sd = load_set_h5(dataset_dir, set_id)
    t = sd["time_s"]
    cons = sd["consensus_df"]
    available_lifts = sorted(int(x) for x in cons["lift_id"].unique())
    chosen_lifts = [lid for lid in _parse_lift_ids(lift_ids, available_lifts) if lid in available_lifts]
    if not chosen_lifts:
        raise ValueError(f"No matching lifts found in {available_lifts} for '{lift_ids}'")

    if isinstance(sensor_codes, str):
        chosen_sensors = [s.strip() for s in sensor_codes.split(",") if s.strip()]
    else:
        chosen_sensors = [str(s).strip() for s in sensor_codes if str(s).strip()]
    for sc in chosen_sensors:
        if sc not in sd["sensors"]:
            raise ValueError(f"Sensor '{sc}' not found in {sd['sensors']}")

    if isinstance(axes, str):
        axes_list = [a.strip() for a in axes.split(",") if a.strip()]
    else:
        axes_list = list(axes)
    axis_idx_map = {"X": 0, "Y": 1, "Z": 2}
    selected_axes = [ax.upper() for ax in axes_list if ax.upper() in axis_idx_map]
    if signal_mode in {"axes", "both"} and not selected_axes:
        selected_axes = ["X", "Y", "Z"]

    order_map = {m: i for i, m in enumerate(MARKER_ORDER)}
    lift_m = cons[cons["lift_id"].isin(chosen_lifts)].copy()
    lift_m["_ord"] = lift_m["marker_name"].map(order_map)
    lift_m = lift_m.sort_values(["lift_id", "_ord"]).reset_index(drop=True)

    t0 = float(lift_m["time_s"].min()) - 0.2
    t1 = float(lift_m["time_s"].max()) + 0.2
    mask = (t >= t0) & (t <= t1)
    t_win = t[mask]

    mods = ["accel", "gyro"] if modality == "both" else [modality]
    n_rows = len(mods)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08)

    color_i = 0
    for row_i, mod in enumerate(mods, start=1):
        prefix = "A" if mod == "accel" else "G"
        unit = "m/s²" if mod == "accel" else "deg/s"
        mod_label = "Acceleration" if mod == "accel" else "Angular Velocity"

        for sc in chosen_sensors:
            s_idx = sd["sensors"].index(sc)
            arr = (sd["accel"] if mod == "accel" else sd["gyro"])[mask, s_idx, :]

            if signal_mode in {"axes", "both"}:
                for ax in selected_axes:
                    ax_i = axis_idx_map[ax]
                    color = _PALETTE[color_i % len(_PALETTE)]
                    color_i += 1
                    fig.add_trace(
                        go.Scatter(
                            x=t_win,
                            y=arr[:, ax_i],
                            mode="lines",
                            name=f"{sc} {prefix}{ax}",
                            line=dict(width=1.4, color=color),
                        ),
                        row=row_i,
                        col=1,
                    )

            if signal_mode in {"magnitude", "both"}:
                mag = np.linalg.norm(arr, axis=1)
                color = _PALETTE[color_i % len(_PALETTE)]
                color_i += 1
                fig.add_trace(
                    go.Scatter(
                        x=t_win,
                        y=mag,
                        mode="lines",
                        name=f"{sc} |{prefix}|",
                        line=dict(width=1.6, color=color, dash="solid" if signal_mode == "magnitude" else "dash"),
                    ),
                    row=row_i,
                    col=1,
                )

        fig.update_yaxes(title_text=f"{mod_label} ({unit})", row=row_i, col=1)

    # Shaded phase bands & markers per selected lift
    show_ann_text = len(chosen_lifts) <= 2
    for lid in chosen_lifts:
        sub_m = lift_m[lift_m["lift_id"] == lid].reset_index(drop=True)
        for i in range(len(sub_m) - 1):
            x_start = float(sub_m.loc[i, "time_s"])
            x_end = float(sub_m.loc[i + 1, "time_s"])
            p_name = str(sub_m.loc[i, "marker_name"])
            for row_i in range(1, n_rows + 1):
                fig.add_vrect(
                    x0=x_start,
                    x1=x_end,
                    fillcolor=_PHASE_BAND_COLORS[i % len(_PHASE_BAND_COLORS)],
                    layer="below",
                    line_width=0,
                    row=row_i,
                    col=1,
                )
                fig.add_vline(x=x_start, line_width=1, line_dash="dot", line_color="#4b5563", row=row_i, col=1)
            if show_ann_text:
                lbl = f"L{lid}:P{i + 1}:{p_name}" if len(chosen_lifts) > 1 else f"P{i + 1}:{p_name}"
                fig.add_annotation(
                    x=0.5 * (x_start + x_end),
                    y=1.01,
                    yref="paper",
                    text=lbl,
                    showarrow=False,
                    textangle=-65,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=8.5, color="#1f2937"),
                )

        last_x = float(sub_m.iloc[-1]["time_s"])
        for row_i in range(1, n_rows + 1):
            fig.add_vline(x=last_x, line_width=1, line_dash="dot", line_color="#4b5563", row=row_i, col=1)

    lifts_lbl = f"Lifts {chosen_lifts[0]}–{chosen_lifts[-1]}" if len(chosen_lifts) > 1 else f"Lift {chosen_lifts[0]}"
    sensors_lbl = ", ".join(chosen_sensors)
    fig.update_layout(
        title=dict(
            text=f"<b>Figure 4: 11-Phase Annotation Scheme ({set_id}, {lifts_lbl}, {sensors_lbl} · {modality})</b>",
            x=0.5, y=0.98, font=dict(size=13),
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=490 if n_rows == 1 else 700,
        margin=dict(l=70, r=25, t=135 if show_ann_text else 70, b=70),
        legend=dict(orientation="h", y=-0.12 if n_rows == 1 else -0.08, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    _pub_axes(fig, rows=n_rows, cols=1)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    return fig
