"""General OpenDatasetStore initialization and default DataFrame display."""

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
    clean_columns: bool = True,
) -> None:
    """Display a DataFrame using the environment's default renderer (Colab / IDE Data Wrangler)."""
    from IPython.display import display

    if df is None or df.empty:
        print("No records found.")
        return

    view_df = df.copy()
    if clean_columns:
        view_df.columns = [str(c).replace("_", " ").title() for c in view_df.columns]

    display(view_df)
