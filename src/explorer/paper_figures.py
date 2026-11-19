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
    "#4E8EC2",  # Pastel Blue
    "#E08B63",  # Pastel Coral
    "#52B79A",  # Pastel Mint
    "#C78EB0",  # Pastel Mauve
    "#E8B866",  # Pastel Amber
    "#7EC2E6",  # Pastel Sky
]
_HEATMAP_SCALE = [[0.0, "#F4A582"], [0.5, "#FAD9A1"], [1.0, "#9AE0C8"]]
_HEATMAP_SCALE_INV = [[0.0, "#9AE0C8"], [0.4, "#FAD9A1"], [1.0, "#F4A582"]]
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


def _static_window_mask(sd: dict) -> np.ndarray:
    """Return boolean mask of static (pre_rest / post_rest) samples for a Set."""
    n = len(sd["time_s"])
    mask = np.zeros(n, dtype=bool)
    cons = sd["consensus_df"]
    if not cons.empty:
        for _, grp in cons.groupby("lift_id"):
            m_dict = dict(zip(grp["marker_name"], grp["imu_idx"]))
            if "start" in m_dict and "setup" in m_dict:
                s, e = int(m_dict["start"]), int(m_dict["setup"])
                if e > s + 10:
                    mask[s:e] = True
            if "post_rest" in m_dict and "end" in m_dict:
                s, e = int(m_dict["post_rest"]), int(m_dict["end"])
                if e > s + 10:
                    mask[s:e] = True
    if not np.any(mask):
        mask[: min(100, n)] = True
    return mask


def _compute_dataset_imu_metrics(dataset_dir: str | Path) -> dict:
    """Compute per-(set, sensor) quality metrics across all 54 Sets x 26 IMUs."""
    tables = load_summary_tables(dataset_dir)
    set_ids = sorted(tables["sets"]["set_id"].tolist())

    records: list[dict] = []
    completeness_matrix: list[list[float]] = []
    sensor_names: list[str] | None = None
    set_lags_ms: list[float] = []

    for sid in set_ids:
        sd = load_set_h5(dataset_dir, sid)
        if sensor_names is None:
            sensor_names = sd["sensors"]

        acc = sd["accel"]
        gyr = sd["gyro"]
        fs = float(sd["metadata"].get("imu_fs_hz", 100.0))
        static_mask = _static_window_mask(sd)
        n = acc.shape[0]
        row_comp: list[float] = []

        # Reference spine dynamic accel magnitude for inter-IMU sync lag
        us_idx = sensor_names.index("US") if "US" in sensor_names else 0
        us_acc = acc[:, us_idx, :]
        us_ok = not np.all(np.isnan(us_acc)) and not any(np.nanstd(us_acc[:, j]) == 0 for j in range(3))
        if us_ok:
            us_mag = np.linalg.norm(pd.DataFrame(us_acc).ffill().bfill().fillna(0.0).to_numpy(), axis=1)
            us_dyn = us_mag - np.median(us_mag)
            us_dyn = (us_dyn - np.mean(us_dyn)) / (np.std(us_dyn) + 1e-8)

        for s_idx, code in enumerate(sensor_names):
            a_sl = acc[:, s_idx, :]
            g_sl = gyr[:, s_idx, :]

            valid_mask = (~np.any(np.isnan(a_sl), axis=1)) & (~np.any(np.isnan(g_sl), axis=1))
            if np.any(valid_mask) and (
                any(np.nanstd(a_sl[:, j]) == 0 for j in range(3))
                or any(np.nanstd(g_sl[:, j]) == 0 for j in range(3))
            ):
                comp_pct = 0.0
            else:
                comp_pct = float(valid_mask.mean() * 100.0) if n else 0.0
            row_comp.append(comp_pct)

            missing_pct = 100.0 - comp_pct
            acc_clip_pct = float((np.abs(a_sl) > 80.0).any(axis=1).mean() * 100.0) if n else 0.0
            gyr_clip_pct = float((np.abs(g_sl) > 2000.0).any(axis=1).mean() * 100.0) if n else 0.0

            static_acc_norm = np.nan
            static_gyr_bias = np.nan
            if comp_pct > 50.0:
                a_stat = a_sl[static_mask]
                g_stat = g_sl[static_mask]
                a_mags = np.linalg.norm(a_stat, axis=1)
                a_mags = a_mags[~np.isnan(a_mags)]
                if len(a_mags):
                    static_acc_norm = float(np.mean(a_mags))
                g_mags = np.linalg.norm(g_stat, axis=1)
                g_mags = g_mags[~np.isnan(g_mags)]
                if len(g_mags):
                    static_gyr_bias = float(np.mean(g_mags))

                # Inter-IMU lag vs US (ms) within ±200 ms window
                if us_ok and code != "US":
                    s_mag = np.linalg.norm(pd.DataFrame(a_sl).ffill().bfill().fillna(0.0).to_numpy(), axis=1)
                    s_dyn = s_mag - np.median(s_mag)
                    s_dyn = (s_dyn - np.mean(s_dyn)) / (np.std(s_dyn) + 1e-8)
                    max_lag = int(round(0.2 * fs))
                    corr = np.correlate(us_dyn, s_dyn, mode="full")
                    mid = len(us_dyn) - 1
                    win = corr[mid - max_lag : mid + max_lag + 1]
                    lag_samples = int(np.argmax(win) - max_lag)
                    set_lags_ms.append(lag_samples * (1000.0 / fs))

            records.append(
                {
                    "set_id": sid,
                    "sensor_code": code,
                    "completeness_pct": comp_pct,
                    "missing_pct": missing_pct,
                    "static_acc_norm": static_acc_norm,
                    "static_gyr_bias": static_gyr_bias,
                    "acc_clip_pct": acc_clip_pct,
                    "gyr_clip_pct": gyr_clip_pct,
                }
            )
        completeness_matrix.append(row_comp)

    return {
        "df": pd.DataFrame(records),
        "set_ids": set_ids,
        "sensor_names": sensor_names or [],
        "completeness_matrix": completeness_matrix,
        "lags_ms": np.array(set_lags_ms, dtype=float),
    }


def build_imu_quality_table(dataset_dir: str | Path) -> pd.DataFrame:
    """Option 1: Summary quality table across all 1,404 (54 Sets x 26 IMUs) recordings."""
    data = _compute_dataset_imu_metrics(dataset_dir)
    df = data["df"]
    lags = np.abs(data["lags_ms"])

    def _summarize(series: pd.Series | np.ndarray, threshold_lbl: str, pass_fn, name: str) -> dict:
        arr = pd.Series(series).dropna()
        if arr.empty:
            return {"Metric": name, "Expected / Threshold": threshold_lbl, "Mean ± SD": "NA", "Median [IQR]": "NA", "Min – Max": "NA", "Recordings Passing (%)": "NA"}
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        pct_pass = float(pass_fn(arr).mean() * 100.0)
        return {
            "Metric": name,
            "Expected / Threshold": threshold_lbl,
            "Mean ± SD": f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}",
            "Median [IQR]": f"{q50:.2f} [{q25:.2f}, {q75:.2f}]",
            "Min – Max": f"{arr.min():.2f} – {arr.max():.2f}",
            "Recordings Passing (%)": f"{pct_pass:.1f}%",
        }

    rows = [
        _summarize(df["completeness_pct"], "> 99%", lambda s: s > 99.0, "Data completeness (% valid samples)"),
        _summarize(df["missing_pct"], "< 1%", lambda s: s < 1.0, "Missing / flatlined samples (%)"),
        _summarize(df["static_acc_norm"], "9.81 ± 0.50 m/s²", lambda s: np.abs(s - 9.81) <= 0.50, "Static accelerometer norm (m/s²)"),
        _summarize(df["static_gyr_bias"], "< 5.0 deg/s", lambda s: s < 5.0, "Static gyroscope bias magnitude (deg/s)"),
        _summarize(df["acc_clip_pct"], "≈ 0% (< 0.1%)", lambda s: s < 0.1, "Accelerometer saturation (% > 80 m/s²)"),
        _summarize(df["gyr_clip_pct"], "≈ 0% (< 0.1%)", lambda s: s < 0.1, "Gyroscope saturation (% > 2000 deg/s)"),
        _summarize(lags, "< 10 ms (1 sample)", lambda s: s <= 10.0, "Inter-IMU dynamic lag vs. US (ms)"),
    ]

    comp_arr = df["completeness_pct"].dropna()
    acc_arr = df["static_acc_norm"].dropna()
    gyr_arr = df["static_gyr_bias"].dropna()
    print(
        f'📝 Draft text: "Across all {len(df):,} sensor recordings ({len(data["set_ids"])} sets × {len(data["sensor_names"])} IMUs), '
        f'data completeness was {comp_arr.mean():.1f} ± {comp_arr.std(ddof=1):.1f}% (min {comp_arr.min():.1f}%). '
        f'The accelerometer norm during static periods was {acc_arr.mean():.2f} ± {acc_arr.std(ddof=1):.2f} m/s², '
        f'and static gyroscope bias was {gyr_arr.mean():.2f} ± {gyr_arr.std(ddof=1):.2f} deg/s."'
    )
    return pd.DataFrame(rows)


def plot_imu_validation_figure(dataset_dir: str | Path) -> go.Figure:
    """Option 2: Compact 4-panel technical validation figure across all 54 Sets x 26 IMUs."""
    data = _compute_dataset_imu_metrics(dataset_dir)
    df = data["df"]
    sensors = data["sensor_names"]

    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.18,
        horizontal_spacing=0.09,
        subplot_titles=(
            "<b>(a) Data Completeness (% Valid Samples)</b>",
            "<b>(b) Static Accelerometer Norm per IMU</b>",
            "<b>(c) Static Gyroscope Bias per IMU</b>",
            "<b>(d) Inter-IMU Synchronization Lag vs. US</b>",
        ),
    )

    # Panel (a): Completeness heatmap
    fig.add_trace(
        go.Heatmap(
            z=data["completeness_matrix"],
            x=sensors,
            y=data["set_ids"],
            colorscale=_HEATMAP_SCALE,
            zmin=0.0,
            zmax=100.0,
            colorbar=dict(title="% Valid", len=0.40, y=0.80, x=0.45),
            hovertemplate="<b>%{y} · %{x}</b><br>Valid: %{z:.1f}%<extra></extra>",
        ),
        row=1, col=1,
    )

    # Panel (b): Static accelerometer norm boxplots (26 IMUs)
    for code in sensors:
        vals = df[df["sensor_code"] == code]["static_acc_norm"].dropna()
        fig.add_trace(
            go.Box(y=vals, name=code, marker_color=_PALETTE[0], showlegend=False, boxpoints="outliers"),
            row=1, col=2,
        )
    fig.add_hline(y=9.81, line_dash="dash", line_color="#D55E00", annotation_text="9.81 m/s²", row=1, col=2)

    # Panel (c): Static gyroscope bias boxplots (26 IMUs)
    for code in sensors:
        vals = df[df["sensor_code"] == code]["static_gyr_bias"].dropna()
        fig.add_trace(
            go.Box(y=vals, name=code, marker_color=_PALETTE[2], showlegend=False, boxpoints="outliers"),
            row=2, col=1,
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#374151", row=2, col=1)

    # Panel (d): Inter-IMU synchronization lag histogram (ms)
    if len(data["lags_ms"]):
        fig.add_trace(
            go.Histogram(
                x=data["lags_ms"],
                nbinsx=25,
                marker_color=_PALETTE[1],
                marker_line=dict(width=0.8, color="#111827"),
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="#111827", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="<b>Technical Validation of IMU Data (54 Sets × 26 Sensors = 1,404 Recordings)</b>",
            x=0.5, y=0.98, font=dict(size=14),
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=820,
        margin=dict(l=70, r=25, t=75, b=75),
    )
    _pub_axes(fig, rows=2, cols=2)
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=8), row=1, col=1)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=7), row=1, col=1)
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=8), row=1, col=2)
    fig.update_yaxes(title_text="Static |a| (m/s²)", row=1, col=2)
    fig.update_xaxes(title_text="IMU Sensor", tickangle=-60, tickfont=dict(size=8), row=2, col=1)
    fig.update_yaxes(title_text="Static |ω| (deg/s)", row=2, col=1)
    fig.update_xaxes(title_text="Lag vs. US (ms)", row=2, col=2)
    fig.update_yaxes(title_text="Recordings (count)", row=2, col=2)
    return fig


_BLUE_TO_RED_SCALE = [
    [0.00, "#EBF5FB"],
    [0.25, "#AED6F1"],
    [0.55, "#5DADE2"],
    [0.80, "#FADBD8"],
    [0.93, "#F1948A"],
    [1.00, "#E74C3C"],
]


def plot_set_annotator_grid(dataset_dir: str | Path, set_id: str = "set_0001") -> go.Figure:
    """Plot a single Set's annotator frames (uncolored) + SD (ms) rows (colored blue-to-red)."""
    sd = load_set_h5(dataset_dir, set_id)
    ann_df = sd["annotators_df"]
    cons_df = sd["consensus_df"]
    if ann_df.empty:
        raise ValueError(f"No annotator records in {set_id}")

    markers = [m for m in MARKER_ORDER if m in set(ann_df["marker_name"])]
    mean_map = {
        (int(r["lift_id"]), r["marker_name"]): float(r["mean_frame"])
        for _, r in cons_df.iterrows()
    }
    sd_ms_map = {
        (int(r["lift_id"]), r["marker_name"]): float(r["sd_ms"])
        for _, r in cons_df.iterrows()
    }

    y_labels: list[str] = []
    z_vals: list[list[float]] = []
    text_vals: list[list[str]] = []

    for lid in sorted(int(x) for x in ann_df["lift_id"].unique()):
        sub = ann_df[ann_df["lift_id"] == lid]
        for aid in sorted(sub["annotator_id"].unique()):
            y_labels.append(f"L{lid} · {aid}")
            row_txt = []
            sub_a = sub[sub["annotator_id"] == aid].set_index("marker_name")
            for m in markers:
                row_txt.append(str(int(sub_a.loc[m, "frame"])) if m in sub_a.index else "NA")
            z_vals.append([np.nan] * len(markers))
            text_vals.append(row_txt)

        # Uncolored Mean row right before SD
        y_labels.append(f"L{lid} · Mean")
        z_vals.append([np.nan] * len(markers))
        text_vals.append([
            f"{mean_map[(lid, m)]:.1f}" if (lid, m) in mean_map else "NA"
            for m in markers
        ])

        # Only color the SD (ms) row
        y_labels.append(f"L{lid} · SD")
        row_sd = [sd_ms_map.get((lid, m), np.nan) for m in markers]
        z_vals.append(row_sd)
        text_vals.append([f"{v:.0f}" if not np.isnan(v) else "NA" for v in row_sd])

    zmax = 300.0

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=markers,
            y=y_labels,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=9, color="#111827"),
            colorscale=_BLUE_TO_RED_SCALE,
            zmin=0.0,
            zmax=zmax,
            xgap=2,
            ygap=2,
            colorbar=dict(title="SD (ms)"),
            hovertemplate="<b>%{y} · %{x}</b><br>Value: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>Annotator Frames, Mean & Inter-Annotator SD (ms) — {set_id}</b>", x=0.5, font=dict(size=14)),
        xaxis_title="Phase Marker",
        yaxis_title="Lift · Annotator / Mean / SD (ms)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#f1f5f9",
        font=_FONT,
        height=max(460, 26 * len(y_labels) + 140),
        margin=dict(l=110, r=35, t=60, b=90),
    )
    _pub_axes(fig)
    fig.update_xaxes(tickangle=-30, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def plot_all_sets_annotation_sd_heatmap(dataset_dir: str | Path) -> go.Figure:
    """Heatmap of inter-annotator SD (ms) across all (set_id, lift_id) x 12 phase markers."""
    m_df = load_summary_tables(dataset_dir)["phase_markers"].copy()
    if m_df.empty:
        return go.Figure()

    m_df["set_lift"] = m_df["set_id"].astype(str) + " · L" + m_df["lift_id"].astype(str)
    markers = [m for m in MARKER_ORDER if m in set(m_df["marker_name"])]
    pivot = m_df.pivot(index="set_lift", columns="marker_name", values="sd_ms")[markers]
    order_df = m_df[["set_id", "lift_id", "set_lift"]].drop_duplicates().sort_values(["set_id", "lift_id"])
    pivot = pivot.reindex(order_df["set_lift"].tolist())

    z_arr = pivot.to_numpy()
    zmax = 300.0

    fig = go.Figure(
        data=go.Heatmap(
            z=z_arr,
            x=list(pivot.columns),
            y=list(pivot.index),
            texttemplate="%{z:.0f}",
            textfont=dict(size=8, color="#111827"),
            colorscale=_BLUE_TO_RED_SCALE,
            zmin=0.0,
            zmax=zmax,
            xgap=1,
            ygap=1,
            colorbar=dict(title="SD (ms)"),
            hovertemplate="<b>%{y} · %{x}</b><br>SD: %{z:.1f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Inter-Annotator SD (ms) Across All Sets & Lifts</b>", x=0.5, font=dict(size=14)),
        xaxis_title="Phase Marker",
        yaxis_title="Set · Lift",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=max(1400, 18 * len(pivot.index) + 180),
        margin=dict(l=120, r=35, t=65, b=95),
    )
    _pub_axes(fig)
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=8))
    return fig


def build_annotation_agreement_table(
    dataset_dir: str | Path,
    threshold_ms: float = 200.0,
) -> pd.DataFrame:
    """Table: Per-marker and overall inter-annotator agreement (right-skewed summary)."""
    m_df = load_summary_tables(dataset_dir)["phase_markers"].copy()
    if m_df.empty:
        return pd.DataFrame()

    present = [m for m in MARKER_ORDER if m in set(m_df["marker_name"])]

    def _row_stats(label: str, sub: pd.DataFrame) -> dict:
        n_total = len(sub)
        all3 = sub[sub["n_annotators"] >= 3]
        pct_all3 = (len(all3) / n_total * 100.0) if n_total else 0.0
        sd_vals = all3["sd_ms"].dropna().to_numpy(dtype=float)
        if len(sd_vals) == 0:
            return {
                "Marker type": label,
                "n instances": n_total,
                "Marked by all 3 (%)": f"{pct_all3:.1f}%",
                "Median SD (ms) [IQR]": "NA",
                "Mean ± SD (ms)": "NA",
                "90th pct (ms)": "NA",
                "95th pct (ms)": "NA",
                f"% < {int(threshold_ms)} ms": "NA",
                "Max (ms)": "NA",
            }
        q25, med, q75, p90, p95 = np.percentile(sd_vals, [25, 50, 75, 90, 95])
        pct_lt = float((sd_vals < threshold_ms).mean() * 100.0)
        return {
            "Marker type": label,
            "n instances": n_total,
            "Marked by all 3 (%)": f"{pct_all3:.1f}%",
            "Median SD (ms) [IQR]": f"{med:.1f} [{q25:.1f}, {q75:.1f}]",
            "Mean ± SD (ms)": f"{np.mean(sd_vals):.1f} ± {np.std(sd_vals, ddof=1):.1f}",
            "90th pct (ms)": f"{p90:.1f}",
            "95th pct (ms)": f"{p95:.1f}",
            f"% < {int(threshold_ms)} ms": f"{pct_lt:.1f}%",
            "Max (ms)": f"{np.max(sd_vals):.1f}",
            "_med": med,
        }

    rows = [_row_stats(m, m_df[m_df["marker_name"] == m]) for m in present]
    overall = _row_stats("All markers", m_df)
    rows.append(overall)

    # Print ready-to-paste manuscript text
    all3_all = m_df[m_df["n_annotators"] >= 3]["sd_ms"].dropna().to_numpy(dtype=float)
    if len(all3_all):
        q25, med, q75, p90 = np.percentile(all3_all, [25, 50, 75, 90])
        pct3 = len(all3_all) / len(m_df) * 100.0
        marker_rows = [r for r in rows[:-1] if "_med" in r]
        best_m = min(marker_rows, key=lambda r: r["_med"])
        worst_m = max(marker_rows, key=lambda r: r["_med"])
        print(
            f'📝 Draft text: "Of {len(m_df):,} marker instances, {pct3:.1f}% were identified by all three annotators. '
            f'For these, the inter-annotator SD of marker timestamps was right-skewed, with a median of {med:.1f} ms '
            f'[IQR {q25:.1f}–{q75:.1f} ms]; 90% of instances showed an SD below {p90:.1f} ms '
            f'(≈ {p90 / 10.0:.1f} samples at 100 Hz). Agreement was highest for {best_m["Marker type"]} '
            f'(median {best_m["_med"]:.1f} ms) and lowest for {worst_m["Marker type"]} '
            f'(median {worst_m["_med"]:.1f} ms)."'
        )

    out_df = pd.DataFrame(rows).drop(columns=["_med"], errors="ignore")
    return out_df


def plot_annotation_agreement_figure(
    dataset_dir: str | Path,
    log_x_ecdf: bool = False,
) -> go.Figure:
    """Two-panel agreement figure: (a) pooled SD histogram with median & 95th pct, (b) ECDF per marker."""
    m_df = load_summary_tables(dataset_dir)["phase_markers"].copy()
    all3 = m_df[m_df["n_annotators"] >= 3].dropna(subset=["sd_ms"])
    sd_all = all3["sd_ms"].to_numpy(dtype=float)

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.10,
        subplot_titles=(
            "<b>(a) Pooled Inter-Annotator SD Distribution</b>",
            "<b>(b) Cumulative Distribution (ECDF) per Marker</b>",
        ),
    )

    if len(sd_all):
        med = float(np.median(sd_all))
        p95 = float(np.percentile(sd_all, 95))

        fig.add_trace(
            go.Histogram(
                x=sd_all,
                nbinsx=40,
                marker_color=_PALETTE[0],
                marker_line=dict(width=0.7, color="#111827"),
                name="All markers",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_vline(
            x=med, line_width=2, line_dash="solid", line_color="#D55E00",
            annotation_text=f"Median: {med:.1f} ms", annotation_position="top right",
            row=1, col=1,
        )
        fig.add_vline(
            x=p95, line_width=2, line_dash="dash", line_color="#111827",
            annotation_text=f"95th pct: {p95:.1f} ms", annotation_position="top right",
            row=1, col=1,
        )

    present = [m for m in MARKER_ORDER if m in set(all3["marker_name"])]
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    for idx, marker in enumerate(present):
        vals = np.sort(all3[all3["marker_name"] == marker]["sd_ms"].to_numpy(dtype=float))
        if len(vals) == 0:
            continue
        x_vals = np.maximum(vals, 1.0) if log_x_ecdf else vals
        y_vals = np.arange(1, len(vals) + 1) / len(vals) * 100.0
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=marker,
                line=dict(
                    width=1.6,
                    color=_PALETTE[idx % len(_PALETTE)],
                    dash=dash_styles[(idx // len(_PALETTE)) % len(dash_styles)],
                ),
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title=dict(
            text="<b>Inter-Annotator Agreement of Event Markers (3 Annotators/Set, 54 Sets)</b>",
            x=0.5, y=0.98, font=dict(size=14),
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=470,
        margin=dict(l=65, r=25, t=75, b=85),
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center", font=dict(size=10)),
    )
    _pub_axes(fig, rows=1, cols=2)
    fig.update_xaxes(title_text="Inter-Annotator SD (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Marker Instances (count)", row=1, col=1)
    fig.update_xaxes(
        title_text="Inter-Annotator SD (ms)" + (" [log]" if log_x_ecdf else ""),
        type="log" if log_x_ecdf else "linear",
        row=1, col=2,
    )
    fig.update_yaxes(title_text="Cumulative Instances (%)", range=[0, 102], row=1, col=2)
    return fig
