"""Interactive ipywidgets UI for single-annotator rep-phase segmentation."""

from __future__ import annotations

import json

import pandas as pd
from IPython.display import display

from wearable_biomechanics.annotation.player import PHASES, new_rep_template, tag_for


def launch_rep_annotator(
    store,
    entry_type: str,
    entry_id: str,
    annotator_id: int,
) -> None:
    """Display the interactive Rep Phase Annotator widget bound to `store`."""
    import ipywidgets as widgets
    state = {
        "annotator_id": annotator_id,
        "current_rep_idx": 0,
        "annotations": [],
        "is_locked": False,
        "is_updating_ui": False,
    }

    rep_selector = widgets.Dropdown(
        options=[("Rep 1", 0)],
        value=0,
        description="Active Rep:",
        layout=widgets.Layout(width="200px"),
    )
    btn_add_rep = widgets.Button(
        description="➕ Add Rep", button_style="success", layout=widgets.Layout(width="120px")
    )
    btn_del_rep = widgets.Button(
        description="🗑️ Delete Rep", button_style="danger", layout=widgets.Layout(width="130px")
    )
    btn_reload = widgets.Button(description="🔄 Reload", layout=widgets.Layout(width="110px"))
    btn_save_store = widgets.Button(
        description="💾 Save to Store", button_style="primary", layout=widgets.Layout(width="160px")
    )
    btn_unlock = widgets.Button(
        description="🔓 Unlock to Edit", button_style="warning", layout=widgets.Layout(width="160px")
    )
    comment_input = widgets.Text(
        value="", placeholder="Optional comment", description="Comment:", layout=widgets.Layout(width="300px")
    )
    status_label = widgets.HTML(value="")

    json_preview = widgets.Textarea(
        value="", description="", layout=widgets.Layout(width="460px", height="460px")
    )

    def refresh_preview():
        json_preview.value = json.dumps(state["annotations"], indent=2)

    phase_inputs, phase_rows = {}, []
    for phase in PHASES:
        int_box = widgets.IntText(
            value=0,
            description=phase,
            style={"description_width": "140px"},
            layout=widgets.Layout(width="340px"),
        )

        def _make_cb(p: str):
            def _cb(change):
                if state["is_updating_ui"]:
                    return
                idx = state["current_rep_idx"]
                if 0 <= idx < len(state["annotations"]):
                    state["annotations"][idx][p] = int(change["new"] or 0)
                    refresh_preview()

            return _cb

        int_box.observe(_make_cb(phase), names="value")
        phase_inputs[phase] = int_box
        phase_rows.append(int_box)

    lock_summary = widgets.HTML(value="")
    lock_banner = widgets.HBox(
        [lock_summary, btn_unlock],
        layout=widgets.Layout(
            display="none",
            justify_content="space-between",
            align_items="center",
            padding="16px",
            background="#1e293b",
            border="1px solid #38bdf8",
            border_radius="8px",
            margin="10px 0",
        ),
    )

    control_bar = widgets.HBox(
        [
            rep_selector,
            btn_add_rep,
            btn_del_rep,
            widgets.HTML("<span style='border-left:1px solid #3f3f46; margin:0 8px; height:24px;'></span>"),
            comment_input,
            btn_save_store,
            btn_reload,
        ],
        layout=widgets.Layout(align_items="center", margin="0 0 10px 0"),
    )

    left_panel = widgets.VBox([widgets.HTML("<b>Phase Frame Indices:</b>")] + phase_rows)
    right_panel = widgets.VBox([widgets.HTML("<b>Live JSON Preview / Quick Edit:</b>"), json_preview])
    editor_container = widgets.VBox(
        [control_bar, widgets.HBox([left_panel, right_panel], layout=widgets.Layout(gap="24px"))]
    )

    def set_ui_locked(locked: bool):
        state["is_locked"] = locked
        for box in phase_inputs.values():
            box.disabled = locked
        for w in (rep_selector, btn_add_rep, btn_del_rep, btn_reload, btn_save_store, comment_input, json_preview):
            w.disabled = locked
        editor_container.layout.display = "none" if locked else "flex"
        lock_banner.layout.display = "flex" if locked else "none"

    def switch_to_rep(idx: int):
        reps = state["annotations"]
        if not reps:
            return
        idx = max(0, min(idx, len(reps) - 1))
        state["current_rep_idx"] = idx
        state["is_updating_ui"] = True
        try:
            rep_selector.unobserve(on_rep_selected, names="value")
            try:
                rep_selector.options = [(f"Rep {i + 1}", i) for i in range(len(reps))]
                rep_selector.value = idx
            finally:
                rep_selector.observe(on_rep_selected, names="value")

            data = reps[idx]
            for p in PHASES:
                phase_inputs[p].value = int(data.get(p, 0))
        finally:
            state["is_updating_ui"] = False
        refresh_preview()

    def load_from_store():
        tag = tag_for(state["annotator_id"])
        try:
            df = store.get_entry_processed_data(
                entry_type=entry_type, entry_id=entry_id, tag=tag
            )
        except Exception:
            df = None

        if df is not None and not df.empty:
            cols = [c for c in PHASES if c in df.columns]
            state["annotations"] = df[cols].to_dict(orient="records")
            status_label.value = (
                f"<span style='color:#38bdf8;'>📂 Loaded {len(state['annotations'])} rep(s) "
                f"for Experiment <b>{entry_id}</b> (tag <code>{tag}</code>).</span>"
            )
        else:
            state["annotations"] = [new_rep_template()]
            status_label.value = (
                f"<span style='color:#a1a1aa;'>📝 No prior data for <b>{entry_id}</b>. "
                f"Initialized 1 blank rep.</span>"
            )
        switch_to_rep(0)

    def on_rep_selected(change):
        if change["new"] is not None and not state["is_updating_ui"]:
            switch_to_rep(change["new"])

    def on_add_rep(btn):
        orig = btn.description
        btn.disabled, btn.description = True, "⏳ Adding..."
        try:
            state["annotations"].append(new_rep_template())
            new_idx = len(state["annotations"]) - 1
            switch_to_rep(new_idx)
            status_label.value = (
                f"<span style='color:#38bdf8;'>➕ Added and switched to <b>Rep {new_idx + 1}</b>.</span>"
            )
        finally:
            btn.description, btn.disabled = orig, False

    def on_del_rep(btn):
        if len(state["annotations"]) <= 1:
            status_label.value = "<span style='color:#fbbf24;'>⚠️ At least 1 rep must remain.</span>"
            return
        orig = btn.description
        btn.disabled, btn.description = True, "⏳ Deleting..."
        try:
            removed_idx = state["current_rep_idx"]
            state["annotations"].pop(removed_idx)
            new_idx = max(0, removed_idx - 1)
            switch_to_rep(new_idx)
            status_label.value = (
                f"<span style='color:#f87171;'>🗑️ Removed Rep {removed_idx + 1}. "
                f"Switched to <b>Rep {new_idx + 1}</b>.</span>"
            )
        finally:
            btn.description, btn.disabled = orig, False

    def on_reload(btn):
        orig = btn.description
        btn.disabled, btn.description = True, "⏳ Reloading..."
        try:
            load_from_store()
            status_label.value = "<span style='color:#38bdf8;'>🔄 Reloaded from store (unsaved edits discarded).</span>"
        finally:
            btn.description, btn.disabled = orig, False

    def on_unlock(btn):
        orig = btn.description
        btn.disabled, btn.description = True, "⏳ Unlocking..."
        try:
            set_ui_locked(False)
            status_label.value = "<span style='color:#fbbf24;'>🔓 Editor unlocked for modification.</span>"
        finally:
            btn.description, btn.disabled = orig, False

    def on_save(btn):
        orig = btn.description
        btn.disabled, btn.description = True, "⏳ Saving..."
        status_label.value = "<span style='color:#94a3b8;'>⏳ Writing annotations to store...</span>"
        aid = state["annotator_id"]

        try:
            parsed = json.loads(json_preview.value)
            if isinstance(parsed, list) and parsed:
                state["annotations"] = parsed
        except Exception as e:
            status_label.value = f"<span style='color:#f87171;'>❌ Invalid JSON: {e}</span>"
            btn.description, btn.disabled = orig, False
            return

        rows = []
        for i, rep in enumerate(state["annotations"]):
            row = {"annotator_id": aid, "rep": i + 1}
            for p in PHASES:
                row[p] = int(rep.get(p, 0))
            rows.append(row)
        df = pd.DataFrame(rows)[["annotator_id", "rep"] + PHASES]
        tag = tag_for(aid)

        try:
            existing = store.get_entry_processed_data(
                entry_type=entry_type, entry_id=entry_id, tag=tag
            )
        except Exception:
            existing = None

        try:
            if existing is not None and not existing.empty:
                store.replace_processed_data(entry_type, entry_id, tag, df)
                verb = "Replaced"
            else:
                store.add_processed_data(
                    entry_type=entry_type,
                    entry_id=entry_id,
                    df=df,
                    tag=tag,
                    comment=comment_input.value or f"annotator={aid}",
                )
                verb = "Saved"

            lock_summary.value = (
                f"<div style='font-size:14px; color:#f4f4f5;'>"
                f"✅ <b>Successfully {verb}!</b> Saved <b>{len(df)} rep(s)</b> for Experiment "
                f"<code style='color:#38bdf8; font-size:15px;'>{entry_id}</code> "
                f"(Annotator #{aid} → <code>{tag}</code>).<br/>"
                f"<span style='color:#94a3b8; font-size:12px; margin-top:4px; display:inline-block;'>"
                f"The editor is now locked to protect this data. To move to the next experiment, "
                f"select it in the video cell and re-run.</span></div>"
            )
            set_ui_locked(True)
            status_label.value = ""
        except Exception as e:
            status_label.value = f"<span style='color:#f87171;'>❌ Store error: {e}</span>"
        finally:
            btn.description, btn.disabled = orig, False

    rep_selector.observe(on_rep_selected, names="value")
    btn_add_rep.on_click(on_add_rep)
    btn_del_rep.on_click(on_del_rep)
    btn_reload.on_click(on_reload)
    btn_save_store.on_click(on_save)
    btn_unlock.on_click(on_unlock)

    header = widgets.HTML(
        f"<div style='display:flex; justify-content:space-between; align-items:center; "
        f"padding:10px 16px; background:#18181b; color:#f4f4f5; border-radius:8px; "
        f"margin-bottom:8px; border:1px solid #27272a; font-family:system-ui,sans-serif;'>"
        f"<div><span style='color:#94a3b8; font-size:11px; text-transform:uppercase;'>"
        f"Target Experiment ID</span><br/>"
        f"<b style='color:#38bdf8; font-size:17px;'>🔬 {entry_id}</b> "
        f"<span style='color:#71717a; font-size:12px; margin-left:8px;'>"
        f"type: <code style='color:#fbbf24;'>{entry_type}</code></span></div>"
        f"<div style='text-align:right;'><span style='color:#94a3b8; font-size:11px; text-transform:uppercase;'>"
        f"Annotator</span><br/><b style='color:#4ade80; font-size:16px;'>ID #{annotator_id}</b></div></div>"
    )

    main_ui = widgets.VBox([header, status_label, lock_banner, editor_container])
    load_from_store()
    display(main_ui)
