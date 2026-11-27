"""Sync root-level summary CSVs into per-Set HDF5 archives and remove /fault_log."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_DATASET_DIR = (
    "/content/drive/MyDrive/Academic/Research/Wearable Sensing/Wearable Biomechanics/Clean&JerkDataset"
)


def _h5_scalar(val):
    """Convert pandas/numpy scalar values to h5py-safe native Python types."""
    if pd.isna(val):
        return ""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return str(val)


def sync_dataset_h5(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    remove_fault_log: bool = True,
    sync_sets: bool = True,
    sync_participants: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    """Update `sets/*.h5` from `sets_summary.csv` and `participant_summary.csv`, and remove `/fault_log`.

    Args:
        dataset_dir: Root directory of `Clean&JerkDataset`.
        remove_fault_log: Delete the `/fault_log` HDF5 group if present.
        sync_sets: Update `/metadata` attributes from `sets_summary.csv`.
        sync_participants: Update `/participant_metadata` attributes from `participant_summary.csv`.
        dry_run: Inspect changes without modifying `.h5` files.

    Returns:
        Summary counts dictionary.
    """
    root = Path(dataset_dir)
    sets_dir = root / "sets"
    if not sets_dir.exists():
        raise FileNotFoundError(f"Sets directory not found: {sets_dir}")

    sets_csv = root / "sets_summary.csv"
    part_csv = root / "participant_summary.csv"

    sets_map: dict[str, dict] = {}
    if sync_sets and sets_csv.exists():
        s_df = pd.read_csv(sets_csv)
        if "set_id" in s_df.columns:
            sets_map = {str(r["set_id"]).strip(): r.to_dict() for _, r in s_df.iterrows()}

    part_map: dict[str, dict] = {}
    if sync_participants and part_csv.exists():
        p_df = pd.read_csv(part_csv)
        if "participant_id" in p_df.columns:
            part_map = {str(r["participant_id"]).strip(): r.to_dict() for _, r in p_df.iterrows()}

    h5_files = sorted(sets_dir.glob("*.h5"))
    stats = {
        "files_scanned": len(h5_files),
        "fault_logs_removed": 0,
        "metadata_synced": 0,
        "participant_metadata_synced": 0,
    }

    mode = "r" if dry_run else "r+"
    for h5_path in h5_files:
        set_id = h5_path.stem
        with h5py.File(h5_path, mode) as f:
            if remove_fault_log and "fault_log" in f:
                if not dry_run:
                    del f["fault_log"]
                stats["fault_logs_removed"] += 1

            if sync_sets and set_id in sets_map:
                g_meta = f["metadata"] if "metadata" in f else (f.create_group("metadata") if not dry_run else None)
                if g_meta is not None:
                    if not dry_run:
                        for k, v in sets_map[set_id].items():
                            g_meta.attrs[str(k)] = _h5_scalar(v)
                    stats["metadata_synced"] += 1

            # Resolve participant_id from sets_summary.csv, /metadata, or /participant_metadata
            pid = ""
            if set_id in sets_map and sets_map[set_id].get("participant_id"):
                pid = str(sets_map[set_id]["participant_id"]).strip()
            elif "metadata" in f and "participant_id" in f["metadata"].attrs:
                pid = str(f["metadata"].attrs["participant_id"]).strip()
            elif "participant_metadata" in f and "participant_id" in f["participant_metadata"].attrs:
                pid = str(f["participant_metadata"].attrs["participant_id"]).strip()

            if sync_participants and pid and pid in part_map:
                g_pmeta = (
                    f["participant_metadata"]
                    if "participant_metadata" in f
                    else (f.create_group("participant_metadata") if not dry_run else None)
                )
                if g_pmeta is not None:
                    if not dry_run:
                        for k, v in part_map[pid].items():
                            g_pmeta.attrs[str(k)] = _h5_scalar(v)
                    stats["participant_metadata_synced"] += 1

    action = "DRY-RUN" if dry_run else "UPDATED"
    print(
        f"[{action}] Scanned {stats['files_scanned']} .h5 files | "
        f"/fault_log removed: {stats['fault_logs_removed']} | "
        f"/metadata synced: {stats['metadata_synced']} | "
        f"/participant_metadata synced: {stats['participant_metadata_synced']}"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync summary CSVs into sets/*.h5 and remove /fault_log.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Path to Clean&JerkDataset root.")
    parser.add_argument("--keep-fault-log", action="store_true", help="Do not delete /fault_log from .h5 files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without modifying .h5 files.")
    args = parser.parse_args()

    sync_dataset_h5(
        dataset_dir=args.dataset_dir,
        remove_fault_log=not args.keep_fault_log,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

