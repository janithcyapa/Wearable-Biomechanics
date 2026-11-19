"""General OpenDatasetStore initialization and AG Grid interactive table display."""

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
    theme: str = "ag-theme-alpine-dark",
    height: int = 420,
) -> None:
    """Render an interactive AG Grid table (`ipyaggrid`) in dark theme.

    Falls back to standard `IPython.display.display(df)` if `ipyaggrid` is not installed.
    """
    from IPython.display import display

    if df is None or df.empty:
        print("No records found.")
        return

    view_df = df.copy()
    if clean_columns:
        view_df.columns = [str(c).replace("_", " ").title() for c in view_df.columns]

    try:
        from ipyaggrid import Grid

        grid_options = {
            "columnDefs": [{"field": c} for c in view_df.columns],
            "defaultColDef": {
                "sortable": True,
                "filter": True,
                "resizable": True,
                "floatingFilter": True,
                "minWidth": 110,
            },
            "pagination": True,
            "paginationPageSize": page_length,
            "enableRangeSelection": True,
        }
        grid = Grid(
            grid_data=view_df,
            grid_options=grid_options,
            theme=theme,
            height=height,
            quick_filter=True,
            export_csv=True,
            export_excel=False,
            columns_fit="auto",
        )
        display(grid)
        return
    except ImportError:
        pass

    display(view_df)
