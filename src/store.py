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
    """Render a minimal, interactive DataFrame viewer in Jupyter or Colab.

    Uses `itables` (search, sort, pagination, column visibility, CSV export) when
    installed, falling back to standard IPython display.
    """
    if df is None or df.empty:
        print("No records found.")
        return

    view_df = df.copy()
    if clean_columns:
        view_df.columns = [str(c).replace("_", " ").title() for c in view_df.columns]

    try:
        from itables import show

        show(
            view_df,
            paging=True,
            pageLength=page_length,
            scrollX=True,
            classes="display compact nowrap",
            column_filters="header",
            buttons=["colvis", "csvHtml5"],
        )
        return
    except ImportError:
        pass

    from IPython.display import display

    display(view_df)
