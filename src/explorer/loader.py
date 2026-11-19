"""Loaders for the exported Clean&JerkDataset summary CSVs and per-Set .h5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_CLEAN_DATASET_DIR = (
    "/content/drive/MyDrive/Academic/Research/Wearable Sensing/Wearable Biomechanics/Clean&JerkDataset"
)


def load_summary_tables(dataset_dir: str | Path = DEFAULT_CLEAN_DATASET_DIR) -> dict[str, pd.DataFrame]:
    """Load all root-level summary CSVs from `dataset_dir`."""
    root = Path(dataset_dir)
    files = {
        "participants": "participant_summary.csv",
        "sets": "sets_summary.csv",
        "devices": "device_summary.csv",
        "faults": "fault_log.csv",
        "phase_markers": "phase_markers_summary.csv",
        "data_dictionary": "data_dictionary.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        fpath = root / fname
        tables[key] = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame()
    return tables


def _decode_arr(arr) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]


def load_set_h5(dataset_dir: str | Path, set_id: str) -> dict:
    """Load all groups and arrays from `sets/{set_id}.h5`."""
    h5_path = Path(dataset_dir) / "sets" / f"{set_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Set HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        metadata = dict(f["metadata"].attrs)
        participant_metadata = dict(f["participant_metadata"].attrs)

        sensors = _decode_arr(f["imu_data/sensor_names"][:])
        axes = _decode_arr(f["imu_data/axis_names"][:])
        accel = f["imu_data/accel"][:]
        gyro = f["imu_data/gyro"][:]
        time_s = f["imu_data/time_s"][:]
        sample_idx = f["imu_data/sample_idx"][:]

        imu_cols: dict[str, np.ndarray] = {"sample_idx": sample_idx, "time_s": time_s}
        for s_idx, code in enumerate(sensors):
            for a_idx, ax in enumerate(axes):
                imu_cols[f"{code}_A{ax}"] = accel[:, s_idx, a_idx]
            for a_idx, ax in enumerate(axes):
                imu_cols[f"{code}_G{ax}"] = gyro[:, s_idx, a_idx]
        imu_df = pd.DataFrame(imu_cols)

        consensus_df = pd.DataFrame(
            {
                k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
                for k, v in f["phase_markers/consensus"].items()
            }
        )
        annotators_df = pd.DataFrame(
            {
                k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
                for k, v in f["phase_markers/annotators"].items()
            }
        )
        fault_log_df = pd.DataFrame(
            {
                k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
                for k, v in f["fault_log"].items()
            }
        )

    return {
        "set_id": set_id,
        "h5_path": str(h5_path),
        "metadata": metadata,
        "participant_metadata": participant_metadata,
        "sensors": sensors,
        "axes": axes,
        "accel": accel,
        "gyro": gyro,
        "time_s": time_s,
        "imu_df": imu_df,
        "consensus_df": consensus_df,
        "annotators_df": annotators_df,
        "fault_log_df": fault_log_df,
    }
