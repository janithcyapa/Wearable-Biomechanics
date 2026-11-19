"""Multi-annotator agreement editor, outlier highlighter, and boundary standardizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from .player import PHASES

_COL_W = "94px"
_LABEL_W = "145px"
_GAP_PX = "4px"


def _get_sd_style(val_frames: float) -> str:
    if pd.isna(val_frames):
        return "background:#444; color:#ccc;"
    if val_frames > 30.0:
        return "background:#cc0000; color:white; font-weight:bold;"
    if val_frames > 8.0:
        return "background:#fde725; color:black; font-weight:bold;"
    if val_frames > 5.0:
        return "background:#5ec962; color:black; font-weight:bold;"
    return "background:#3e4989; color:white;"


def launch_annotation_checker(
    store,
    entry_type: str,
    entry_id: str,
    max_annotators: int = 15,
) -> None:
    """Display the multi-annotator comparison, outlier detection, and save UI."""
    import ipywidgets as widgets
    candidate_tags = [f"rep_segments_a{i}" for i in range(1, max_annotators + 1)]
    raw_dfs: dict[str, pd.DataFrame] = {}
    annotator_tags: list[str] = []

    for tag in candidate_tags:
        try:
            df = store.get_entry_processed_data(
                entry_type=entry_type, entry_id=entry_id, tag=tag
            )
            if df is not None and not df.empty:
                df = df.copy()
                valid_mask = (df[PHASES[1:]] != 0).any(axis=1)
                df = df[valid_mask].sort_values("rep").reset_index(drop=True)
                if not df.empty:
                    if "annotator_id" not in df.columns:
                        df.insert(0, "annotator_id", int(tag.split("_a")[-1]))
                    raw_dfs[tag] = df
                    annotator_tags.append(tag)
        except Exception:
            continue

    if not raw_dfs:
        raise ValueError(f"No valid annotations found for {entry_id}.")

    meta_df = store.list_entries(entry_type=entry_type)
    trial_meta = meta_df.loc[meta_df["entry_id"] == entry_id].iloc[0]
    fps = float(trial_meta["video_fps"]) if "video_fps" in trial_meta else 30.0
    ms_per_frame = 1000.0 / fps
    max_frame = int(trial_meta["cropped_video_frames"]) - 1

    all_reps = sorted({rep for df in raw_dfs.values() for rep in df["rep"].unique()})

    display(
        HTML(
            f"""
    <style>
        .grid-row {{ display:flex !important; flex-direction:row !important; align-items:center !important; gap:{_GAP_PX} !important; width:max-content !important; }}
        .ann-cell {{ width:{_COL_W} !important; min-width:{_COL_W} !important; max-width:{_COL_W} !important; margin:0 !important; padding:0 !important; box-sizing:border-box !important; }}
        .ann-cell input {{ width:100% !important; box-sizing:border-box !important; text-align:right !important; font-family:monospace !important; font-size:12px !important; padding:2px 4px !important; background-color:#1e1e1e !important; color:#f0f0f0 !important; border:1px solid #444 !important; border-radius:3px !important; }}
        .ann-outlier-2v1 input {{ background-color:#6b1414 !important; border:2px solid #ff4d4d !important; color:#ffffff !important; font-weight:bold !important; }}
        .ann-outlier-mean input {{ background-color:#663c00 !important; border:2px solid #ffa726 !important; color:#ffffff !important; font-weight:bold !important; }}
        .ann-seq-error input {{ background-color:#4a148c !important; border:2px solid #e040fb !important; color:#ffffff !important; font-weight:bold !important; }}
    </style>
    """
        )
    )

    input_widgets: dict[tuple, widgets.IntText] = {}
    mean_labels: dict[tuple, widgets.HTML] = {}
    sd_labels: dict[tuple, widgets.HTML] = {}
    sd_ms_labels: dict[tuple, widgets.HTML] = {}

    def update_rep_stats(rep):
        seq_errors = set()
        for tag in annotator_tags:
            prev_val = None
            for col in PHASES:
                if (rep, tag, col) in input_widgets:
                    curr_val = input_widgets[(rep, tag, col)].value
                    if prev_val is not None and curr_val < prev_val:
                        seq_errors.add((rep, tag, col))
                    prev_val = curr_val

        for col in PHASES:
            vals_dict = {}
            for tag in annotator_tags:
                if (rep, tag, col) in input_widgets:
                    w = input_widgets[(rep, tag, col)]
                    vals_dict[tag] = w.value
                    w.remove_class("ann-outlier-2v1")
                    w.remove_class("ann-outlier-mean")
                    w.remove_class("ann-seq-error")

            vals = list(vals_dict.values())
            if not vals:
                continue

            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            sd_ms_val = sd_val * ms_per_frame if not pd.isna(sd_val) else np.nan

            mean_labels[(rep, col)].value = (
                f"<div style='width:{_COL_W}; box-sizing:border-box; text-align:right; font-family:monospace; "
                f"font-size:12px; padding:3px 6px; background:#2b2b2b; color:#bbb; border-radius:3px;'>{mean_val:.1f}</div>"
            )
            sd_style = _get_sd_style(sd_val)
            sd_text = f"{sd_val:.1f}" if not pd.isna(sd_val) else "NaN"
            sd_labels[(rep, col)].value = (
                f"<div style='width:{_COL_W}; box-sizing:border-box; text-align:right; font-family:monospace; "
                f"font-size:12px; padding:3px 6px; {sd_style} border-radius:3px;'>{sd_text}</div>"
            )
            sd_ms_text = f"{sd_ms_val:.0f} ms" if not pd.isna(sd_ms_val) else "NaN"
            sd_ms_labels[(rep, col)].value = (
                f"<div style='width:{_COL_W}; box-sizing:border-box; text-align:right; font-family:monospace; "
                f"font-size:11px; padding:3px 6px; {sd_style} opacity:0.9; border-radius:3px;'>{sd_ms_text}</div>"
            )

            outlier_2v1_tag = None
            if len(vals_dict) == 3:
                (t_low, v_low), (_, v_mid), (t_high, v_high) = sorted(
                    vals_dict.items(), key=lambda x: x[1]
                )
                if (v_mid - v_low) <= 8 and (v_high - v_mid) > 10:
                    outlier_2v1_tag = t_high
                elif (v_high - v_mid) <= 8 and (v_mid - v_low) > 10:
                    outlier_2v1_tag = t_low

            for tag, val in vals_dict.items():
                w = input_widgets[(rep, tag, col)]
                if (rep, tag, col) in seq_errors:
                    w.add_class("ann-seq-error")
                elif tag == outlier_2v1_tag:
                    w.add_class("ann-outlier-2v1")
                elif abs(val - mean_val) > 8.0:
                    w.add_class("ann-outlier-mean")

    rows_ui = []
    header_cells = [
        widgets.HTML(
            f"<div style='width:{_LABEL_W}; min-width:{_LABEL_W}; box-sizing:border-box; "
            f"font-weight:bold; color:#fff; padding:4px 6px;'>Rep / Source</div>"
        )
    ] + [
        widgets.HTML(
            f"<div style='width:{_COL_W}; min-width:{_COL_W}; box-sizing:border-box; text-align:right; "
            f"font-weight:bold; color:#9cdcfe; font-size:11px; padding:4px 4px; "
            f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap;' title='{col}'>{col}</div>"
        )
        for col in PHASES
    ]
    header_box = widgets.HBox(
        header_cells, layout=widgets.Layout(border_bottom="2px solid #666", padding="4px 0")
    )
    header_box.add_class("grid-row")
    rows_ui.append(header_box)

    for rep in all_reps:
        rows_ui.append(
            widgets.HTML(
                f"<div style='background:#0e639c; color:white; font-weight:bold; padding:5px 10px; "
                f"margin-top:12px; border-radius:4px; font-size:13px; width:100%; box-sizing:border-box;'>"
                f"🏋️ REPETITION {rep} (Video FPS: {int(fps)} | 1 frame = {ms_per_frame:.1f} ms)</div>"
            )
        )

        for tag in annotator_tags:
            rep_row = raw_dfs[tag][raw_dfs[tag]["rep"] == rep]
            if rep_row.empty:
                rows_ui.append(
                    widgets.HTML(
                        f"<div style='color:#ff6b6b; font-size:12px; padding:4px 8px;'>"
                        f"⚠️ <b>{tag}</b> is missing Rep {rep} entirely!</div>"
                    )
                )
                continue

            row_cells = [
                widgets.HTML(
                    f"<div style='width:{_LABEL_W}; min-width:{_LABEL_W}; box-sizing:border-box; "
                    f"font-family:monospace; font-size:12px; color:#dcdcaa; padding:3px 6px;'>{tag}</div>"
                )
            ]
            for col in PHASES:
                w_int = widgets.IntText(
                    value=int(rep_row[col].values[0]),
                    layout=widgets.Layout(width=_COL_W, height="26px", margin="0px"),
                )
                w_int.add_class("ann-cell")
                w_int.observe(lambda _c, r=rep: update_rep_stats(r), names="value")
                input_widgets[(rep, tag, col)] = w_int
                row_cells.append(w_int)

            ann_box = widgets.HBox(row_cells, layout=widgets.Layout(margin="2px 0"))
            ann_box.add_class("grid-row")
            rows_ui.append(ann_box)

        for label_text, store_dict, margin, border_bottom in [
            ("↳ Mean (Frames)", mean_labels, "3px 0 0 0", None),
            ("↳ Std Dev (Frames)", sd_labels, "2px 0 0 0", None),
            ("↳ Std Dev (ms)", sd_ms_labels, "2px 0 8px 0", "1px solid #444"),
        ]:
            cells = [
                widgets.HTML(
                    f"<div style='width:{_LABEL_W}; min-width:{_LABEL_W}; box-sizing:border-box; "
                    f"font-size:12px; font-weight:bold; color:#ddd; padding:3px 6px;'>{label_text}</div>"
                )
            ]
            for col in PHASES:
                lbl = widgets.HTML(layout=widgets.Layout(margin="0px"))
                store_dict[(rep, col)] = lbl
                cells.append(lbl)
            layout_kw = {"padding": margin} if border_bottom else {"margin": margin}
            if border_bottom:
                layout_kw["border_bottom"] = border_bottom
            box = widgets.HBox(cells, layout=widgets.Layout(**layout_kw))
            box.add_class("grid-row")
            rows_ui.append(box)

        update_rep_stats(rep)

    status_out = widgets.Output()
    btn_std_bounds = widgets.Button(
        description="⚡ Fix Start/End Boundaries",
        button_style="info",
        tooltip="Pins Rep 1 start_rest=0, final end_rep=max_frame, and enforces rep continuity",
        layout=widgets.Layout(width="210px", height="34px"),
    )
    btn_save = widgets.Button(
        description="💾 Save Changes to Store",
        button_style="success",
        tooltip="Replaces processed data in OpenDatasetStore using store.replace_processed_data",
        layout=widgets.Layout(width="210px", height="34px"),
    )

    def on_standardize_clicked(_b):
        with status_out:
            status_out.clear_output()
            for tag in annotator_tags:
                tag_reps = sorted([r for r in all_reps if (r, tag, "start_rest") in input_widgets])
                if not tag_reps:
                    continue
                input_widgets[(tag_reps[0], tag, "start_rest")].value = 0
                input_widgets[(tag_reps[-1], tag, "end_rep")].value = max_frame
                for idx in range(1, len(tag_reps)):
                    prev_r, curr_r = tag_reps[idx - 1], tag_reps[idx]
                    input_widgets[(curr_r, tag, "start_rest")].value = input_widgets[(prev_r, tag, "end_rest")].value
                    input_widgets[(curr_r, tag, "setup")].value = input_widgets[(prev_r, tag, "end_rep")].value
            print(
                f"⚡ Standardized start_rest (0), final end_rep ({max_frame}), "
                f"and rep transitions across all annotators!"
            )

    def on_save_clicked(_b):
        with status_out:
            status_out.clear_output()
            for tag in annotator_tags:
                df_updated = raw_dfs[tag].copy()
                for idx, row in df_updated.iterrows():
                    r = row["rep"]
                    for col in PHASES:
                        if (r, tag, col) in input_widgets:
                            df_updated.at[idx, col] = int(input_widgets[(r, tag, col)].value)
                store.replace_processed_data(
                    entry_id=entry_id, entry_type=entry_type, df=df_updated, tag=tag
                )
                raw_dfs[tag] = df_updated
                print(f"✅ Replaced processed data in store -> entry_id='{entry_id}', tag='{tag}'")

    btn_std_bounds.on_click(on_standardize_clicked)
    btn_save.on_click(on_save_clicked)

    legend_html = widgets.HTML(
        """
    <div style='font-size:12px; color:#ccc; margin: 4px 0 8px 0; display:flex; gap:14px; flex-wrap:wrap;'>
        <span><b>Box Alerts:</b></span>
        <span><span style='background:#6b1414; border:1px solid #ff4d4d; padding:1px 6px; border-radius:3px; color:#fff;'>Red</span> = 1 Annotator off vs other 2</span>
        <span><span style='background:#663c00; border:1px solid #ffa726; padding:1px 6px; border-radius:3px; color:#fff;'>Orange</span> = &gt;8 frames (&gt;267ms) from Mean</span>
        <span><span style='background:#4a148c; border:1px solid #e040fb; padding:1px 6px; border-radius:3px; color:#fff;'>Purple</span> = Sequence inversion (&lt; previous phase)</span>
    </div>
    """
    )
    toolbar = widgets.HBox(
        [
            widgets.HTML(f"<h3 style='margin:0 15px 0 0; color:#fff;'>📊 {entry_id} ({len(annotator_tags)} Annotators)</h3>"),
            btn_std_bounds,
            btn_save,
        ],
        layout=widgets.Layout(align_items="center", margin="0 0 6px 0"),
    )
    table_container = widgets.VBox(
        rows_ui, layout=widgets.Layout(overflow_x="auto", padding="8px", border="1px solid #444")
    )
    display(toolbar, legend_html, status_out, table_container)
