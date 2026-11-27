"""Per-annotator inter-annotator SD attribution & distribution comparison tools."""

from __future__ import annotations

import math
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .loader import _decode_arr, load_summary_tables

_FONT = dict(family="Arial, Helvetica, sans-serif", size=11, color="#111827")
_PALETTE = [
    "#4E8EC2",  # Pastel Blue
    "#E08B63",  # Pastel Coral
    "#52B79A",  # Pastel Mint
    "#C78EB0",  # Pastel Mauve
    "#E8B866",  # Pastel Amber
    "#7EC2E6",  # Pastel Sky
    "#D96B6B",  # Soft Crimson
    "#6B9E78",  # Sage Green
    "#9E82C4",  # Lavender
    "#C9975B",  # Ochre
]


def _pub_axes(fig: go.Figure, rows: int = 1, cols: int = 1) -> None:
    kw = dict(
        showline=True,
        linewidth=1.1,
        linecolor="#1f2937",
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor="#1f2937",
        showgrid=True,
        gridcolor="#e5e7eb",
        gridwidth=0.6,
    )
    if rows == 1 and cols == 1:
        fig.update_xaxes(zeroline=False, **kw)
        fig.update_yaxes(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8, **kw)
        return
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(zeroline=False, row=r, col=c, **kw)
            fig.update_yaxes(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=0.8, row=r, col=c, **kw)


def _natural_annotator_key(ann_id: str) -> tuple[str, int, str]:
    m = re.match(r"^([A-Za-z_]*)(\d+)$", str(ann_id))
    if m:
        return (m.group(1), int(m.group(2)), str(ann_id))
    return (str(ann_id), 0, str(ann_id))


def _read_group_df(group: h5py.Group) -> pd.DataFrame:
    return pd.DataFrame(
        {
            k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
            for k, v in group.items()
        }
    )


def load_annotator_marker_records(
    dataset_dir: str | Path,
    min_annotators: int = 3,
) -> dict[str, pd.DataFrame]:
    """Load consensus + per-annotator phase markers from all `sets/*.h5` files (metadata/markers only).

    Computes leave-one-annotator-out (LOAO) SD (`sd_without_ms`), SD inflation
    (`delta_sd_ms = sd_ms - sd_without_ms`), and signed peer deviation
    (`peer_dev_ms = (frame_a - mean(other_frames)) * ms_per_frame`) for every
    annotator contribution on instances with `n_annotators >= min_annotators`.
    """
    root = Path(dataset_dir)
    sets_dir = root / "sets"
    sets_df = load_summary_tables(root).get("sets", pd.DataFrame())

    if not sets_df.empty and "set_id" in sets_df.columns:
        h5_files = [sets_dir / f"{sid}.h5" for sid in sets_df["set_id"].tolist()]
        h5_files = [p for p in h5_files if p.exists()]
    else:
        h5_files = sorted(sets_dir.glob("set_*.h5"))

    consensus_parts: list[pd.DataFrame] = []
    annotator_parts: list[pd.DataFrame] = []

    for h5_path in h5_files:
        set_id = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            if "phase_markers" not in f:
                continue
            pm = f["phase_markers"]
            if "consensus" not in pm or "annotators" not in pm:
                continue

            fps = float(f["metadata"].attrs.get("video_fps", 30.0)) if "metadata" in f else 30.0
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0

            c_df = _read_group_df(pm["consensus"])
            a_df = _read_group_df(pm["annotators"])
            if c_df.empty or a_df.empty:
                continue

            c_df["set_id"] = set_id
            c_df["video_fps"] = fps
            a_df["set_id"] = set_id

            consensus_parts.append(c_df)
            annotator_parts.append(a_df)

    if not consensus_parts or not annotator_parts:
        return {"consensus_df": pd.DataFrame(), "contrib_df": pd.DataFrame()}

    consensus_df = pd.concat(consensus_parts, ignore_index=True)
    annotators_df = pd.concat(annotator_parts, ignore_index=True)

    # Filter to consensus instances meeting minimum annotator threshold
    consensus_df = consensus_df[
        (consensus_df["n_annotators"] >= min_annotators) & consensus_df["sd_ms"].notna()
    ].copy()

    keys = ["set_id", "lift_id", "marker_name"]
    merged = annotators_df.merge(
        consensus_df[keys + ["n_annotators", "mean_frame", "sd_frames", "sd_ms", "video_fps"]],
        on=keys,
        how="inner",
    )

    if merged.empty:
        return {"consensus_df": consensus_df, "contrib_df": pd.DataFrame()}

    # Derive exact ms/frame factor per instance (preferring consensus ratio when sd_frames > 0)
    ms_per_frame = np.where(
        (merged["sd_frames"] > 0) & np.isfinite(merged["sd_ms"] / merged["sd_frames"]),
        merged["sd_ms"] / merged["sd_frames"],
        1000.0 / merged["video_fps"],
    )
    merged["ms_per_frame"] = ms_per_frame

    # Compute Leave-One-Annotator-Out (LOAO) SD and signed peer deviation per instance
    sd_without_arr = np.full(len(merged), np.nan, dtype=float)
    peer_dev_arr = np.full(len(merged), np.nan, dtype=float)

    for _, idx in merged.groupby(keys, sort=False).groups.items():
        pos = merged.index.get_indexer(idx)
        frames = merged["frame"].to_numpy(dtype=float)[pos]
        mpf = merged["ms_per_frame"].to_numpy(dtype=float)[pos]
        k = len(frames)
        if k < 2:
            continue
        for j in range(k):
            others = np.delete(frames, j)
            peer_mean = float(np.mean(others))
            peer_dev_arr[pos[j]] = (frames[j] - peer_mean) * mpf[j]
            if len(others) >= 2:
                sd_without_arr[pos[j]] = float(np.std(others, ddof=1)) * mpf[j]

    merged["sd_without_ms"] = sd_without_arr
    merged["delta_sd_ms"] = merged["sd_ms"].to_numpy(dtype=float) - sd_without_arr
    merged["peer_dev_ms"] = peer_dev_arr
    merged["abs_peer_dev_ms"] = np.abs(peer_dev_arr)

    return {"consensus_df": consensus_df, "contrib_df": merged}


def _resolve_records(
    source: str | Path | dict[str, pd.DataFrame],
    min_annotators: int = 3,
) -> dict[str, pd.DataFrame]:
    if isinstance(source, dict) and "consensus_df" in source and "contrib_df" in source:
        return source
    return load_annotator_marker_records(source, min_annotators=min_annotators)


def build_annotator_sd_attribution_table(
    source: str | Path | dict[str, pd.DataFrame],
    min_annotators: int = 3,
    threshold_ms: float = 200.0,
) -> pd.DataFrame:
    """Summary table comparing each annotator's contributed SD distribution & LOAO impact against Overall."""
    rec = _resolve_records(source, min_annotators=min_annotators)
    c_df = rec["consensus_df"]
    m_df = rec["contrib_df"]
    if c_df.empty or m_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for ann_id, sub in m_df.groupby("annotator_id", sort=False):
        sd_vals = sub["sd_ms"].dropna().to_numpy(dtype=float)
        if len(sd_vals) == 0:
            continue
        q25, med, q75, p95 = np.percentile(sd_vals, [25, 50, 75, 95])
        loao_sd = sub["sd_without_ms"].dropna().to_numpy(dtype=float)
        delta_sd = sub["delta_sd_ms"].dropna().to_numpy(dtype=float)
        peer_dev = sub["peer_dev_ms"].dropna().to_numpy(dtype=float)
        abs_dev = sub["abs_peer_dev_ms"].dropna().to_numpy(dtype=float)

        rows.append(
            {
                "Annotator": ann_id,
                "Sets": int(sub["set_id"].nunique()),
                "n instances": int(len(sd_vals)),
                "Median SD (ms) [IQR]": f"{med:.1f} [{q25:.1f}, {q75:.1f}]",
                "Mean ± SD (ms)": f"{np.mean(sd_vals):.1f} ± {np.std(sd_vals, ddof=1):.1f}",
                "95th pct (ms)": f"{p95:.1f}",
                f"% < {int(threshold_ms)} ms": f"{float((sd_vals < threshold_ms).mean() * 100.0):.1f}%",
                "Median SD w/o Ann (ms)": f"{np.median(loao_sd):.1f}" if len(loao_sd) else "NA",
                "Median ΔSD (ms)": f"{np.median(delta_sd):+.1f}" if len(delta_sd) else "NA",
                "Mean ΔSD (ms)": f"{np.mean(delta_sd):+.1f}" if len(delta_sd) else "NA",
                "Mean Peer Bias (ms)": f"{np.mean(peer_dev):+.1f}" if len(peer_dev) else "NA",
                "Median |Peer Dev| (ms)": f"{np.median(abs_dev):.1f}" if len(abs_dev) else "NA",
                "_sort_med": med,
                "_sort_delta": float(np.mean(delta_sd)) if len(delta_sd) else 0.0,
            }
        )

    rows.sort(key=lambda r: (-r["_sort_med"], -r["_sort_delta"], _natural_annotator_key(r["Annotator"])))

    sd_all = c_df["sd_ms"].dropna().to_numpy(dtype=float)
    q25, med, q75, p95 = np.percentile(sd_all, [25, 50, 75, 95])
    rows.append(
        {
            "Annotator": "Overall (Pooled)",
            "Sets": int(c_df["set_id"].nunique()),
            "n instances": int(len(sd_all)),
            "Median SD (ms) [IQR]": f"{med:.1f} [{q25:.1f}, {q75:.1f}]",
            "Mean ± SD (ms)": f"{np.mean(sd_all):.1f} ± {np.std(sd_all, ddof=1):.1f}",
            "95th pct (ms)": f"{p95:.1f}",
            f"% < {int(threshold_ms)} ms": f"{float((sd_all < threshold_ms).mean() * 100.0):.1f}%",
            "Median SD w/o Ann (ms)": "—",
            "Median ΔSD (ms)": "—",
            "Mean ΔSD (ms)": "—",
            "Mean Peer Bias (ms)": "—",
            "Median |Peer Dev| (ms)": "—",
            "_sort_med": -1.0,
            "_sort_delta": 0.0,
        }
    )

    return pd.DataFrame(rows).drop(columns=["_sort_med", "_sort_delta"])


def plot_annotator_sd_small_multiples(
    source: str | Path | dict[str, pd.DataFrame],
    bin_width_ms: float = 20.0,
    min_annotators: int = 3,
    x_max_ms: float | None = None,
    cols: int = 3,
) -> go.Figure:
    """Small-multiples grid: each annotator's contributed SD histogram overlaid against Overall pooled SD."""
    rec = _resolve_records(source, min_annotators=min_annotators)
    c_df = rec["consensus_df"]
    m_df = rec["contrib_df"]
    if c_df.empty or m_df.empty:
        return go.Figure()

    sd_all = c_df["sd_ms"].dropna().to_numpy(dtype=float)
    overall_med = float(np.median(sd_all))
    overall_p95 = float(np.percentile(sd_all, 95))

    if x_max_ms is None:
        x_max_ms = float(math.ceil(max(np.percentile(sd_all, 99.5), overall_p95 * 1.4) / bin_width_ms) * bin_width_ms)

    # Sort annotators by median contributed SD descending
    ann_stats = (
        m_df.groupby("annotator_id")["sd_ms"]
        .agg(["count", "median"])
        .reset_index()
    )
    ann_stats["nat_key"] = ann_stats["annotator_id"].map(_natural_annotator_key)
    ann_stats = ann_stats.sort_values(by=["median", "nat_key"], ascending=[False, True])
    annotators = ann_stats["annotator_id"].tolist()

    n_ann = len(annotators)
    n_cols = max(1, min(cols, n_ann))
    n_rows = int(math.ceil(n_ann / n_cols))

    # Build clean bin edges shared across every subplot for exact apples-to-apples comparison
    bin_edges = np.arange(0.0, x_max_ms + bin_width_ms, bin_width_ms)
    bin_centers = bin_edges[:-1] + bin_width_ms / 2.0
    overall_counts, _ = np.histogram(np.clip(sd_all, 0.0, x_max_ms - 1e-9), bins=bin_edges)
    overall_prob = overall_counts / max(len(sd_all), 1)

    titles = []
    for _, row in ann_stats.iterrows():
        ann = row["annotator_id"]
        med = float(row["median"])
        n_i = int(row["count"])
        diff = med - overall_med
        titles.append(f"<b>{ann}</b> (n={n_i:,} | Med={med:.1f} ms, {diff:+.1f})")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.055,
        vertical_spacing=max(0.06, min(0.12, 0.55 / max(n_rows, 1))),
    )

    for idx, ann in enumerate(annotators):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        sub_sd = m_df.loc[m_df["annotator_id"] == ann, "sd_ms"].dropna().to_numpy(dtype=float)
        ann_med = float(np.median(sub_sd))
        ann_counts, _ = np.histogram(np.clip(sub_sd, 0.0, x_max_ms - 1e-9), bins=bin_edges)
        ann_prob = ann_counts / max(len(sub_sd), 1)

        color = "#D96B6B" if ann_med > overall_med + 5.0 else _PALETTE[0]
        show_leg = idx == 0

        # Background: Overall pooled SD distribution (gray step/bar)
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=overall_prob,
                width=bin_width_ms,
                marker_color="rgba(156, 163, 175, 0.35)",
                marker_line=dict(width=0.6, color="rgba(107, 114, 128, 0.65)"),
                name="Overall (all annotators)",
                legendgroup="overall",
                showlegend=show_leg,
                hovertemplate="Overall<br>SD bin: %{x:.0f} ms<br>Proportion: %{y:.3f}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        # Foreground: This annotator's contributed SD distribution
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=ann_prob,
                mode="lines+markers",
                line=dict(color=color, width=2.2, shape="hvh"),
                marker=dict(size=3.5, color=color),
                fill="tozeroy",
                fillcolor="rgba(217, 107, 107, 0.16)" if ann_med > overall_med + 5.0 else "rgba(78, 142, 194, 0.18)",
                name="Annotator contributed",
                legendgroup="annotator",
                showlegend=show_leg,
                hovertemplate=f"{ann}<br>SD bin: %{{x:.0f}} ms<br>Proportion: %{{y:.3f}}<extra></extra>",
            ),
            row=r,
            col=c,
        )

        # Vertical lines: Overall median (gray dashed) vs Annotator median (colored solid)
        fig.add_vline(
            x=overall_med,
            line_width=1.4,
            line_dash="dash",
            line_color="#4B5563",
            row=r,
            col=c,
        )
        fig.add_vline(
            x=ann_med,
            line_width=1.8,
            line_dash="solid",
            line_color=color,
            row=r,
            col=c,
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Per-Annotator Contributed SD Distribution vs. Overall Pooled Distribution</b>"
                f"<br><sup>Sorted by median SD (descending) | Dashed gray = Overall median ({overall_med:.1f} ms), "
                "Solid line = Annotator median (red if > +5 ms above overall)</sup>"
            ),
            x=0.5,
            y=0.985,
            font=dict(size=14),
        ),
        barmode="overlay",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=max(340, 235 * n_rows + 95),
        margin=dict(l=65, r=25, t=95, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5),
    )

    _pub_axes(fig, rows=n_rows, cols=n_cols)
    fig.update_xaxes(range=[0, x_max_ms])
    for c in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Inter-Annotator SD (ms)", row=n_rows, col=c)
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Proportion", row=r, col=1)

    return fig


def plot_annotator_sd_ecdf_and_box(
    source: str | Path | dict[str, pd.DataFrame],
    min_annotators: int = 3,
    x_max_ms: float | None = None,
) -> go.Figure:
    """Two-panel compact comparison: (A) per-annotator SD boxplots vs Overall, (B) overlaid ECDFs."""
    rec = _resolve_records(source, min_annotators=min_annotators)
    c_df = rec["consensus_df"]
    m_df = rec["contrib_df"]
    if c_df.empty or m_df.empty:
        return go.Figure()

    sd_all = np.sort(c_df["sd_ms"].dropna().to_numpy(dtype=float))
    overall_med = float(np.median(sd_all))
    overall_p95 = float(np.percentile(sd_all, 95))
    if x_max_ms is None:
        x_max_ms = float(np.percentile(sd_all, 99.0))

    order = (
        m_df.groupby("annotator_id")["sd_ms"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>(A) Contributed Inter-Annotator SD by Annotator</b>",
            "<b>(B) Empirical CDF: Individual Annotators vs. Overall</b>",
        ],
        horizontal_spacing=0.09,
    )

    # Panel A: Overall + each annotator boxplot
    fig.add_trace(
        go.Box(
            y=sd_all,
            name="Overall",
            marker_color="#6B7280",
            line_color="#1F2937",
            boxmean=True,
            boxpoints=False,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for idx, ann in enumerate(order):
        sub_sd = m_df.loc[m_df["annotator_id"] == ann, "sd_ms"].dropna().to_numpy(dtype=float)
        color = _PALETTE[idx % len(_PALETTE)]
        fig.add_trace(
            go.Box(
                y=sub_sd,
                name=ann,
                marker_color=color,
                boxmean=True,
                boxpoints=False,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=overall_med,
        line_dash="dash",
        line_color="#111827",
        line_width=1.4,
        annotation_text=f"Overall median ({overall_med:.1f} ms)",
        annotation_position="top right",
        row=1,
        col=1,
    )

    # Panel B: Overlaid ECDFs
    for idx, ann in enumerate(order):
        sub_sd = np.sort(m_df.loc[m_df["annotator_id"] == ann, "sd_ms"].dropna().to_numpy(dtype=float))
        y_ecdf = np.arange(1, len(sub_sd) + 1) / len(sub_sd)
        color = _PALETTE[idx % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=sub_sd,
                y=y_ecdf,
                mode="lines",
                line=dict(width=1.7, color=color),
                opacity=0.85,
                name=f"{ann} (med {np.median(sub_sd):.1f})",
                hovertemplate=f"{ann}<br>SD ≤ %{{x:.1f}} ms: %{{y:.1%}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Bold black Overall ECDF on top
    y_all = np.arange(1, len(sd_all) + 1) / len(sd_all)
    fig.add_trace(
        go.Scatter(
            x=sd_all,
            y=y_all,
            mode="lines",
            line=dict(width=3.2, color="#111827"),
            name=f"Overall (med {overall_med:.1f})",
            hovertemplate="Overall<br>SD ≤ %{x:.1f} ms: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text="<b>Direct Annotator SD Comparison (Distribution & Cumulative Agreement)</b>",
            x=0.5,
            y=0.97,
            font=dict(size=14),
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=470,
        margin=dict(l=65, r=25, t=75, b=60),
        legend=dict(font=dict(size=9.5)),
    )
    _pub_axes(fig, rows=1, cols=2)
    fig.update_xaxes(title_text="Annotator", tickangle=-35, row=1, col=1)
    fig.update_yaxes(title_text="Inter-Annotator SD (ms)", range=[0, x_max_ms], row=1, col=1)
    fig.update_xaxes(title_text="Inter-Annotator SD (ms)", range=[0, x_max_ms], row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Proportion (ECDF)", range=[0, 1.02], row=1, col=2)
    return fig


def plot_annotator_loao_impact(
    source: str | Path | dict[str, pd.DataFrame],
    min_annotators: int = 3,
) -> go.Figure:
    """Two-panel causal attribution: (A) Leave-One-Annotator-Out ΔSD, (B) Signed peer deviation (bias vs jitter)."""
    rec = _resolve_records(source, min_annotators=min_annotators)
    m_df = rec["contrib_df"]
    if m_df.empty:
        return go.Figure()

    stats = (
        m_df.groupby("annotator_id")
        .agg(
            mean_delta=("delta_sd_ms", "mean"),
            med_delta=("delta_sd_ms", "median"),
            n=("delta_sd_ms", "count"),
        )
        .reset_index()
        .sort_values(by="mean_delta", ascending=False)
    )
    order = stats["annotator_id"].tolist()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>(A) Leave-One-Annotator-Out SD Impact (ΔSD = SD<sub>all</sub> − SD<sub>−a</sub>)</b>",
            "<b>(B) Signed Deviation from Peers (t<sub>a</sub> − mean(t<sub>−a</sub>))</b>",
        ],
        horizontal_spacing=0.10,
    )

    # Panel A: Mean and Median ΔSD bars/markers
    bar_colors = ["#D96B6B" if v > 0 else "#52B79A" for v in stats["mean_delta"]]
    fig.add_trace(
        go.Bar(
            x=stats["annotator_id"],
            y=stats["mean_delta"],
            marker_color=bar_colors,
            marker_line=dict(width=0.8, color="#1f2937"),
            name="Mean ΔSD (ms)",
            hovertemplate="%{x}<br>Mean ΔSD: %{y:+.2f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=stats["annotator_id"],
            y=stats["med_delta"],
            mode="markers",
            marker=dict(symbol="diamond", size=8, color="#111827", line=dict(width=1, color="white")),
            name="Median ΔSD (ms)",
            hovertemplate="%{x}<br>Median ΔSD: %{y:+.2f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.0, line_color="#111827", line_width=1.2, row=1, col=1)

    # Panel B: Signed peer deviation boxplots
    for idx, ann in enumerate(order):
        sub_dev = m_df.loc[m_df["annotator_id"] == ann, "peer_dev_ms"].dropna().to_numpy(dtype=float)
        color = "#D96B6B" if float(stats.loc[stats["annotator_id"] == ann, "mean_delta"].iloc[0]) > 0 else "#4E8EC2"
        fig.add_trace(
            go.Box(
                y=sub_dev,
                name=ann,
                marker_color=color,
                boxmean=True,
                boxpoints=False,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.add_hline(y=0.0, line_color="#111827", line_width=1.2, line_dash="dash", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Causal Annotator Attribution: Did an Annotator Inflate SD or Just Annotate Harder Sets?</b>"
                "<br><sup>Positive ΔSD (red) = removing this annotator lowers SD | Negative ΔSD (green) = this annotator aligns closely with peers</sup>"
            ),
            x=0.5,
            y=0.97,
            font=dict(size=13.5),
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        height=470,
        margin=dict(l=65, r=25, t=85, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.02),
    )
    _pub_axes(fig, rows=1, cols=2)
    fig.update_xaxes(title_text="Annotator (sorted by Mean ΔSD)", tickangle=-35, row=1, col=1)
    fig.update_yaxes(title_text="ΔSD = SD(with a) − SD(without a) [ms]", row=1, col=1)
    fig.update_xaxes(title_text="Annotator", tickangle=-35, row=1, col=2)
    fig.update_yaxes(title_text="Signed Peer Deviation [ms] (+ late / − early)", row=1, col=2)
    return fig
