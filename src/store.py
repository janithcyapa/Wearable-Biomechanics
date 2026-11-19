"""General OpenDatasetStore initialization and minimal interactive table display."""

from __future__ import annotations

import pandas as pd

DEFAULT_DATASET_PATH = "/content/drive/MyDrive/Datasets/Wearable-Biomechanics"


def init_store(dataset_path: str = DEFAULT_DATASET_PATH):
    """Initialize OpenDatasetStore with project-standard ID and filename formats."""
    from open_dataset_store import OpenDatasetStore

    return OpenDatasetStore(
        base_dir=dataset_path,
        backend="local",
        entity_id_format="sub_{num:04d}",
        entry_id_format="{type}_{num:04d}",
        raw_filename_format="{entry_id}_{entity_id}_{ts}.csv",
        processed_filename_format="{entry_id}_{tag}.parquet",
    )


def display_table(
    df: pd.DataFrame,
    page_length: int = 15,
    clean_columns: bool = True,
) -> None:
    """Render a minimal, theme-native interactive DataFrame viewer (search, sort, paginate).

    Uses pure `ipywidgets` + native `IPython.display.display(df)` so it always
    matches the notebook's dark or light theme cleanly.
    """
    from IPython.display import display

    if df is None or df.empty:
        print("No records found.")
        return

    view_df = df.copy()
    if clean_columns:
        view_df.columns = [str(c).replace("_", " ").title() for c in view_df.columns]

    try:
        import ipywidgets as widgets
    except ImportError:
        display(view_df)
        return

    cols = list(view_df.columns)
    state = {"page": 0}

    search_box = widgets.Text(
        value="",
        placeholder="Search all columns...",
        description="Search:",
        layout=widgets.Layout(width="260px"),
    )
    sort_col = widgets.Dropdown(
        options=["(original)"] + cols,
        value="(original)",
        description="Sort:",
        layout=widgets.Layout(width="210px"),
    )
    sort_asc = widgets.ToggleButton(
        value=True,
        description="↑ Asc",
        layout=widgets.Layout(width="80px"),
    )
    page_size = widgets.Dropdown(
        options=[10, 15, 25, 50, ("All", len(view_df))],
        value=page_length if page_length in (10, 15, 25, 50) else 15,
        description="Rows:",
        layout=widgets.Layout(width="135px"),
    )
    btn_prev = widgets.Button(description="◀", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="▶", layout=widgets.Layout(width="40px"))
    page_info = widgets.Label(value="")
    out = widgets.Output()

    def _filtered_sorted() -> pd.DataFrame:
        res = view_df
        q = search_box.value.strip().lower()
        if q:
            mask = res.astype(str).apply(lambda col: col.str.lower().str.contains(q, regex=False)).any(axis=1)
            res = res[mask]
        if sort_col.value != "(original)" and sort_col.value in res.columns:
            res = res.sort_values(by=sort_col.value, ascending=sort_asc.value, kind="mergesort")
        return res

    def _render(_=None):
        res = _filtered_sorted()
        n = len(res)
        ps = max(1, int(page_size.value or n or 1))
        total_pages = max(1, (n + ps - 1) // ps)
        state["page"] = max(0, min(state["page"], total_pages - 1))
        start = state["page"] * ps
        end = min(start + ps, n)

        sort_asc.description = "↑ Asc" if sort_asc.value else "↓ Desc"
        page_info.value = f"{start + 1 if n else 0}–{end} of {n} (Page {state['page'] + 1}/{total_pages})"
        btn_prev.disabled = state["page"] <= 0
        btn_next.disabled = state["page"] >= total_pages - 1

        with out:
            out.clear_output(wait=True)
            display(res.iloc[start:end])

    def _on_filter_change(_change):
        state["page"] = 0
        _render()

    def _on_prev(_b):
        state["page"] = max(0, state["page"] - 1)
        _render()

    def _on_next(_b):
        state["page"] += 1
        _render()

    search_box.observe(_on_filter_change, names="value")
    sort_col.observe(_on_filter_change, names="value")
    sort_asc.observe(_on_filter_change, names="value")
    page_size.observe(_on_filter_change, names="value")
    btn_prev.on_click(_on_prev)
    btn_next.on_click(_on_next)

    controls = widgets.HBox(
        [search_box, sort_col, sort_asc, page_size, btn_prev, btn_next, page_info],
        layout=widgets.Layout(align_items="center", flex_flow="row wrap", gap="6px", margin="0 0 6px 0"),
    )
    display(widgets.VBox([controls, out]))
    _render()
