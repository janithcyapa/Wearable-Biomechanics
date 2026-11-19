"""Batch and incremental exporter from raw OpenDatasetStore files to Clean&JerkDataset."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .nomenclature import (
    LEGACY_TO_STANDARD_MARKERS,
    SENSOR_ORDER,
    STANDARD_MARKERS,
    build_data_dictionary_df,
    build_device_summary_df,
    standardize_imu_arrays,
)

FAULT_LOG_COLUMNS = [
    "fault_id",
    "set_id",
    "sensor_code",
    "fault_type",
    "severity",
    "start_imu_idx",
    "end_imu_idx",
    "start_s",
    "end_s",
    "description",
]


def _load_json(path: str | Path) -> dict:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _clean_float(val, zero_is_nan: bool = False) -> float:
    if val is None or val == "" or str(val).strip() in {"-", "NA", "N/A", "N\\A", "00"}:
        return float("nan")
    try:
        num = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if zero_is_nan and num == 0.0:
        return float("nan")
    return num


def _clean_height_m(val) -> float:
    h = _clean_float(val, zero_is_nan=True)
    if np.isnan(h):
        return h
    return round(h / 100.0, 4) if h > 3.0 else round(h, 4)


def load_and_deduplicate_entries(
    index_dir: str | Path,
    existing_mapping_csv: str | Path | None = None,
) -> tuple[list[dict], dict[str, str], dict[str, dict]]:
    """Load complete + fault_tolerant manifests, merge duplicates, sort oldest-first, and assign fresh IDs."""
    index_dir = Path(index_dir)
    ct = {
        k: v
        for k, v in _load_json(index_dir / "entries_complete_trials.json").items()
        if not k.startswith("__")
    }
    ft = {
        k: v
        for k, v in _load_json(index_dir / "entries_fault_tolerant_trials.json").items()
        if not k.startswith("__")
    }
    subjects = {
        k: v
        for k, v in _load_json(index_dir / "entities_subjects.json").items()
        if not k.startswith("__")
    }

    # Group by original_experiment_id (fallback: timestamp) to catch duplicates
    grouped: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for tier, entries in [("complete_trials", ct), ("fault_tolerant_trials", ft)]:
        for entry_id, meta in entries.items():
            dedup_key = meta.get("original_experiment_id") or f"ts_{meta.get('timestamp')}"
            grouped[str(dedup_key)].append((tier, entry_id, meta))

    merged_records: list[dict] = []
    for _key, items in grouped.items():
        # Prefer complete_trials over fault_tolerant_trials, then lowest entry_id; drop duplicates
        items.sort(key=lambda x: (0 if x[0] == "complete_trials" else 1, x[1]))
        primary_tier, primary_id, primary_meta = items[0]
        dup_ids = [dup_id for _t, dup_id, _m in items[1:]]

        rec = dict(primary_meta)
        rec["processed_files"] = dict(primary_meta.get("processed_files", {}))
        rec["_source_tier"] = primary_tier
        rec["_source_entry_id"] = primary_id
        rec["_has_faults"] = primary_tier == "fault_tolerant_trials"
        rec["_merged_duplicate_ids"] = dup_ids
        merged_records.append(rec)

    # Sort chronologically, oldest first
    merged_records.sort(
        key=lambda r: (
            int(r.get("timestamp") or 0),
            str(r.get("date") or ""),
            str(r.get("time") or ""),
            r["_source_entry_id"],
        )
    )

    # Preserve existing IDs if incremental mapping exists
    existing_set_map: dict[str, str] = {}
    existing_part_map: dict[str, str] = {}
    next_set_num = 1
    next_part_num = 1

    if existing_mapping_csv and Path(existing_mapping_csv).exists():
        mdf = pd.read_csv(existing_mapping_csv)
        for _, row in mdf.iterrows():
            existing_set_map[str(row["source_entry_id"])] = str(row["set_id"])
            existing_part_map[str(row["source_entity_id"])] = str(row["participant_id"])
        if not mdf.empty:
            next_set_num = max(int(str(s).split("_")[1]) for s in mdf["set_id"]) + 1
            next_part_num = max(int(str(p).lstrip("P")) for p in mdf["participant_id"]) + 1

    subject_id_map: dict[str, str] = dict(existing_part_map)
    for rec in merged_records:
        src_entry = rec["_source_entry_id"]
        if src_entry in existing_set_map:
            rec["_set_id"] = existing_set_map[src_entry]
        else:
            rec["_set_id"] = f"set_{next_set_num:04d}"
            next_set_num += 1

        src_sub = str(rec.get("entity_id", ""))
        if src_sub not in subject_id_map:
            subject_id_map[src_sub] = f"P{next_part_num:04d}"
            next_part_num += 1
        rec["_participant_id"] = subject_id_map[src_sub]

    return merged_records, subject_id_map, subjects


def build_id_mapping_df(records: list[dict]) -> pd.DataFrame:
    """Build the private audit-trail DataFrame mapping new IDs back to legacy IDs."""
    rows = []
    sample_h5_written = False
    for rec in records:
        rows.append(
            {
                "set_id": rec["_set_id"],
                "participant_id": rec["_participant_id"],
                "source_tier": rec["_source_tier"],
                "source_entry_id": rec["_source_entry_id"],
                "original_experiment_id": rec.get("original_experiment_id", ""),
                "merged_duplicate_ids": ";".join(rec.get("_merged_duplicate_ids", [])),
                "source_entity_id": rec.get("entity_id", ""),
                "timestamp": rec.get("timestamp", ""),
                "session_date": rec.get("date", ""),
            }
        )
    return pd.DataFrame(rows)


def _build_participant_summary(
    subject_id_map: dict[str, str],
    subjects_raw: dict[str, dict],
    existing_csv: Path | None = None,
) -> pd.DataFrame:
    """Build de-identified participant_summary.csv ordered by participant_id."""
    existing_by_pid: dict[str, dict] = {}
    if existing_csv and existing_csv.exists():
        edf = pd.read_csv(existing_csv)
        for _, row in edf.iterrows():
            existing_by_pid[str(row["participant_id"])] = row.to_dict()

    rows = []
    ordered_pairs = sorted(subject_id_map.items(), key=lambda kv: kv[1])
    for src_sub, pid in ordered_pairs:
        if pid in existing_by_pid:
            rows.append(existing_by_pid[pid])
            continue
        meta = subjects_raw.get(src_sub, {})
        age = _clean_float(meta.get("age"), zero_is_nan=True)
        weight = _clean_float(meta.get("weight"), zero_is_nan=True)
        height = _clean_height_m(meta.get("height"))
        bmi = (
            round(weight / (height**2), 2)
            if not np.isnan(weight) and not np.isnan(height) and height > 0
            else float("nan")
        )
        exp = _clean_float(meta.get("experience"), zero_is_nan=False)
        best_w = _clean_float(meta.get("best_weight"), zero_is_nan=True)
        ach = str(meta.get("achievement", "")).strip()
        if ach in {"", "-", "NA", "N/A", "N\\A"}:
            ach = "NA"

        raw_sex = str(meta.get("sex", "")).strip().lower()
        sex = {"male": "M", "m": "M", "female": "F", "f": "F"}.get(
            raw_sex, raw_sex.upper() if raw_sex and raw_sex not in {"-", "na", "n/a", "n\\a"} else "NA"
        )

        raw_notes = str(meta.get("notes", "")).strip()
        notes = "NA" if raw_notes in {"", "-", "NA", "N/A", "N\\A"} else raw_notes

        rows.append(
            {
                "participant_id": pid,
                "name": str(meta.get("name", "")).strip() or "NA",
                "age_years": int(age) if not np.isnan(age) else pd.NA,
                "sex": sex,
                "height_m": height if not np.isnan(height) else pd.NA,
                "body_mass_kg": round(weight, 2) if not np.isnan(weight) else pd.NA,
                "bmi_kg_m2": bmi if not np.isnan(bmi) else pd.NA,
                "experience_years": exp if not np.isnan(exp) else pd.NA,
                "achievement": ach,
                "best_weight_kg": best_w if not np.isnan(best_w) else pd.NA,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _extract_phase_markers(
    rec: dict,
    source_base: Path,
    video_fps: float,
    imu_fs_hz: float,
    n_imu_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all rep_segments_a* parquets for a Set and compute per-annotator & consensus tables."""
    set_id = rec["_set_id"]
    proc_files = rec.get("processed_files", {})
    ann_rows: list[dict] = []

    for tag, rel_path in sorted(proc_files.items()):
        if not tag.startswith("rep_segments_a"):
            continue
        pq_path = source_base / rel_path
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        if df.empty:
            continue

        annotator_id = tag.replace("rep_segments_", "")
        for _, row in df.iterrows():
            lift_id = int(row.get("rep", 1))
            for legacy_col, std_marker in LEGACY_TO_STANDARD_MARKERS.items():
                if legacy_col not in df.columns:
                    continue
                frame = int(row[legacy_col])
                time_s = round(frame / float(video_fps), 6)
                imu_idx = int(np.clip(round(time_s * float(imu_fs_hz)), 0, max(0, n_imu_samples - 1)))
                ann_rows.append(
                    {
                        "set_id": set_id,
                        "lift_id": lift_id,
                        "marker_name": std_marker,
                        "annotator_id": annotator_id,
                        "frame": frame,
                        "imu_idx": imu_idx,
                        "time_s": time_s,
                    }
                )

    ann_df = pd.DataFrame(ann_rows)
    if ann_df.empty:
        cons_cols = [
            "set_id", "lift_id", "marker_name", "n_annotators",
            "mean_frame", "sd_frames", "sd_ms", "frame", "imu_idx", "time_s",
        ]
        return pd.DataFrame(columns=cons_cols), pd.DataFrame(
            columns=["set_id", "lift_id", "marker_name", "annotator_id", "frame", "imu_idx", "time_s"]
        )

    marker_order = {m: i for i, m in enumerate(STANDARD_MARKERS)}
    cons_rows: list[dict] = []
    ms_per_frame = 1000.0 / float(video_fps)

    for (lift_id, marker_name), grp in ann_df.groupby(["lift_id", "marker_name"], sort=False):
        frames = grp["frame"].to_numpy(dtype=np.float64)
        n_ann = len(frames)
        mean_f = float(np.mean(frames))
        sd_f = float(np.std(frames, ddof=1)) if n_ann > 1 else 0.0
        sd_ms = sd_f * ms_per_frame
        final_frame = int(round(mean_f))
        final_time_s = round(final_frame / float(video_fps), 6)
        final_imu_idx = int(
            np.clip(round(final_time_s * float(imu_fs_hz)), 0, max(0, n_imu_samples - 1))
        )
        cons_rows.append(
            {
                "set_id": set_id,
                "lift_id": int(lift_id),
                "marker_name": marker_name,
                "n_annotators": n_ann,
                "mean_frame": round(mean_f, 4),
                "sd_frames": round(sd_f, 4),
                "sd_ms": round(sd_ms, 2),
                "frame": final_frame,
                "imu_idx": final_imu_idx,
                "time_s": final_time_s,
            }
        )

    cons_df = pd.DataFrame(cons_rows)
    cons_df["_m_ord"] = cons_df["marker_name"].map(marker_order)
    cons_df = cons_df.sort_values(["lift_id", "_m_ord"]).drop(columns=["_m_ord"]).reset_index(drop=True)
    return cons_df, ann_df


def _write_set_h5(
    h5_path: Path,
    set_meta: dict,
    part_meta: dict,
    imu_arrays: dict,
    cons_df: pd.DataFrame,
    ann_df: pd.DataFrame,
) -> None:
    """Write a self-contained HDF5 archive for one Set."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(h5_path, "w") as f:
        g_meta = f.create_group("metadata")
        for k, v in set_meta.items():
            g_meta.attrs[k] = "" if pd.isna(v) else v

        g_pmeta = f.create_group("participant_metadata")
        for k, v in part_meta.items():
            g_pmeta.attrs[k] = "" if pd.isna(v) else v

        g_imu = f.create_group("imu_data")
        ds_acc = g_imu.create_dataset("accel", data=imu_arrays["accel"], compression="gzip", compression_opts=4)
        ds_acc.attrs["units"] = "m/s^2"
        ds_acc.attrs["sample_rate_hz"] = float(set_meta.get("imu_fs_hz", 100.0))

        ds_gyr = g_imu.create_dataset("gyro", data=imu_arrays["gyro"], compression="gzip", compression_opts=4)
        ds_gyr.attrs["units"] = "deg/s"
        ds_gyr.attrs["sample_rate_hz"] = float(set_meta.get("imu_fs_hz", 100.0))

        ds_t = g_imu.create_dataset("time_s", data=imu_arrays["time_s"], compression="gzip", compression_opts=4)
        ds_t.attrs["units"] = "s"
        g_imu.create_dataset("sample_idx", data=imu_arrays["sample_idx"], compression="gzip", compression_opts=4)
        g_imu.create_dataset("sensor_names", data=np.array(SENSOR_ORDER, dtype="S4"))
        g_imu.create_dataset("axis_names", data=np.array(["X", "Y", "Z"], dtype="S2"))

        g_pm = f.create_group("phase_markers")
        g_cons = g_pm.create_group("consensus")
        g_cons.create_dataset("lift_id", data=cons_df["lift_id"].to_numpy(dtype=np.int32) if not cons_df.empty else np.empty(0, dtype=np.int32))
        g_cons.create_dataset("marker_name", data=np.array(cons_df["marker_name"].tolist(), dtype=object) if not cons_df.empty else np.empty(0, dtype=object), dtype=str_dtype)
        g_cons.create_dataset("n_annotators", data=cons_df["n_annotators"].to_numpy(dtype=np.int32) if not cons_df.empty else np.empty(0, dtype=np.int32))
        g_cons.create_dataset("mean_frame", data=cons_df["mean_frame"].to_numpy(dtype=np.float64) if not cons_df.empty else np.empty(0, dtype=np.float64))
        g_cons.create_dataset("sd_frames", data=cons_df["sd_frames"].to_numpy(dtype=np.float64) if not cons_df.empty else np.empty(0, dtype=np.float64))
        g_cons.create_dataset("sd_ms", data=cons_df["sd_ms"].to_numpy(dtype=np.float64) if not cons_df.empty else np.empty(0, dtype=np.float64))
        g_cons.create_dataset("frame", data=cons_df["frame"].to_numpy(dtype=np.int32) if not cons_df.empty else np.empty(0, dtype=np.int32))
        g_cons.create_dataset("imu_idx", data=cons_df["imu_idx"].to_numpy(dtype=np.int32) if not cons_df.empty else np.empty(0, dtype=np.int32))
        g_cons.create_dataset("time_s", data=cons_df["time_s"].to_numpy(dtype=np.float64) if not cons_df.empty else np.empty(0, dtype=np.float64))

        g_ann = g_pm.create_group("annotators")
        g_ann.create_dataset("lift_id", data=ann_df["lift_id"].to_numpy(dtype=np.int32) if not ann_df.empty else np.empty(0, dtype=np.int32))
        g_ann.create_dataset("marker_name", data=np.array(ann_df["marker_name"].tolist(), dtype=object) if not ann_df.empty else np.empty(0, dtype=object), dtype=str_dtype)
        g_ann.create_dataset("annotator_id", data=np.array(ann_df["annotator_id"].tolist(), dtype=object) if not ann_df.empty else np.empty(0, dtype=object), dtype=str_dtype)
        g_ann.create_dataset("frame", data=ann_df["frame"].to_numpy(dtype=np.int32) if not ann_df.empty else np.empty(0, dtype=np.int32))
        g_ann.create_dataset("imu_idx", data=ann_df["imu_idx"].to_numpy(dtype=np.int32) if not ann_df.empty else np.empty(0, dtype=np.int32))
        g_ann.create_dataset("time_s", data=ann_df["time_s"].to_numpy(dtype=np.float64) if not ann_df.empty else np.empty(0, dtype=np.float64))

        # Empty fault_log schema ready for population
        g_fl = f.create_group("fault_log")
        for col, dt in [
            ("fault_id", str_dtype), ("set_id", str_dtype), ("sensor_code", str_dtype),
            ("fault_type", str_dtype), ("severity", str_dtype),
            ("start_imu_idx", np.int32), ("end_imu_idx", np.int32),
            ("start_s", np.float64), ("end_s", np.float64),
            ("description", str_dtype),
        ]:
            g_fl.create_dataset(col, data=np.empty(0, dtype=object if dt is str_dtype else dt), dtype=dt)


def _write_readme(output_dir: Path) -> None:
    content = """# Clean & Jerk Wearable Biomechanics Dataset

Synchronized multi-node 26-IMU (100 Hz) and dual-view RGB video (30 fps) dataset of Olympic Clean & Jerk lifts with multi-annotator phase segmentation.

## Directory Layout
- `data_dictionary.csv` — Field-level schema, types, and physical units.
- `participant_summary.csv` — De-identified athlete anthropometrics and experience.
- `sets_summary.csv` — Per-Set recording metadata, sensor health counts, and fault flags.
- `device_summary.csv` — Canonical 26-IMU anatomical mapping (`RKP`..`US`).
- `fault_log.csv` — Sensor/recording fault log.
- `phase_markers_summary.csv` — Consensus 12-marker boundaries and inter-annotator SD per Lift.
- `sets/set_XXXX.h5` — Self-contained HDF5 archive per Set (`/metadata`, `/participant_metadata`, `/imu_data`, `/phase_markers`, `/fault_log`).
- `videos/set_XXXX_front.mp4`, `set_XXXX_side.mp4` — Synchronized RGB videos.
- `checksums.sha256` — SHA-256 verification manifest.

## Quick Start (Python)
```python
import h5py
with h5py.File("sets/set_0001.h5", "r") as f:
    accel = f["imu_data/accel"][:]   # (N, 26, 3) m/s^2
    gyro = f["imu_data/gyro"][:]     # (N, 26, 3) deg/s
    sensors = [s.decode() for s in f["imu_data/sensor_names"][:]]
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _write_checksums(output_dir: Path) -> None:
    lines = []
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256":
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {p.relative_to(output_dir).as_posix()}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_h5(h5_path: str | Path, show_rows: int = 10) -> dict[str, pd.DataFrame]:
    """Inspect an exported .h5 file and display its metadata, IMU columns, phase markers, and fault log."""
    from IPython.display import display

    h5_path = Path(h5_path)
    print(f"📦 HDF5 Inspection: {h5_path} ({h5_path.stat().st_size / 1e6:.2f} MB)")

    def _decode_arr(arr):
        return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]

    with h5py.File(h5_path, "r") as f:
        meta_df = pd.DataFrame([dict(f["metadata"].attrs)])
        part_df = pd.DataFrame([dict(f["participant_metadata"].attrs)])

        # Build readable IMU preview DataFrame with sensor_names + axis_names
        sensors = _decode_arr(f["imu_data/sensor_names"][:])
        axes = _decode_arr(f["imu_data/axis_names"][:])
        n_preview = min(show_rows, f["imu_data/time_s"].shape[0])
        imu_cols = {
            "sample_idx": f["imu_data/sample_idx"][:n_preview],
            "time_s": f["imu_data/time_s"][:n_preview],
        }
        acc = f["imu_data/accel"][:n_preview]
        gyr = f["imu_data/gyro"][:n_preview]
        for s_idx, code in enumerate(sensors):
            for a_idx, ax in enumerate(axes):
                imu_cols[f"{code}_A{ax}"] = acc[:, s_idx, a_idx]
            for a_idx, ax in enumerate(axes):
                imu_cols[f"{code}_G{ax}"] = gyr[:, s_idx, a_idx]
        imu_preview_df = pd.DataFrame(imu_cols)
        print(
            f"  /imu_data: accel={f['imu_data/accel'].shape} ({f['imu_data/accel'].attrs.get('units','')}), "
            f"gyro={f['imu_data/gyro'].shape} ({f['imu_data/gyro'].attrs.get('units','')})"
        )

        cons_df = pd.DataFrame(
            {
                k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
                for k, v in f["phase_markers/consensus"].items()
            }
        )
        ann_df = pd.DataFrame(
            {
                k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
                for k, v in f["phase_markers/annotators"].items()
            }
        )
        fault_df = pd.DataFrame(
            {
                k: (_decode_arr(v[:]) if v.dtype.kind in {"S", "O"} else v[:])
                for k, v in f["fault_log"].items()
            }
        )

    print("\n--- /metadata ---")
    display(meta_df)
    print("\n--- /participant_metadata ---")
    display(part_df)
    print(f"\n--- /imu_data (first {n_preview} rows × {len(imu_preview_df.columns)} cols) ---")
    display(imu_preview_df)
    print(f"\n--- /phase_markers/consensus ({len(cons_df)} rows) ---")
    display(cons_df.head(show_rows))
    print(f"\n--- /phase_markers/annotators ({len(ann_df)} rows) ---")
    display(ann_df.head(show_rows))
    print(f"\n--- /fault_log ({len(fault_df)} rows) ---")
    display(fault_df)

    return {
        "metadata": meta_df,
        "participant_metadata": part_df,
        "imu_preview": imu_preview_df,
        "phase_markers_consensus": cons_df,
        "phase_markers_annotators": ann_df,
        "fault_log": fault_df,
    }


def clean_export_dir(output_dir: str | Path) -> None:
    """Remove all previously exported dataset files inside `output_dir` for a clean rebuild."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        print(f"ℹ️ Export folder does not exist yet: {out_dir}")
        return

    removed = 0
    for sub in ("sets", "videos"):
        sub_path = out_dir / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)
            removed += 1

    for fname in (
        "README.md",
        "data_dictionary.csv",
        "sets_summary.csv",
        "participant_summary.csv",
        "device_summary.csv",
        "fault_log.csv",
        "phase_markers_summary.csv",
        "checksums.sha256",
    ):
        fpath = out_dir / fname
        if fpath.exists():
            fpath.unlink()
            removed += 1

    print(f"🧹 Cleaned {removed} artifact(s) from {out_dir}")


def export_dataset(
    source_base_dir: str | Path,
    output_dir: str | Path,
    mapping_csv_path: str | Path | None = None,
    index_dir: str | Path | None = None,
    copy_videos: bool = True,
    only_new: bool = False,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    """Export all (or newly added) Sets from `source_base_dir` to `output_dir`.

    When `dry_run=True`, performs full validation and logs per-Set diagnostics
    (sensor counts, missing sensors, column samples, video & annotation checks)
    without writing any files.
    """
    source_base = Path(source_base_dir)
    out_dir = Path(output_dir)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sets").mkdir(exist_ok=True)
        (out_dir / "videos").mkdir(exist_ok=True)

    idx_path = Path(index_dir) if index_dir else source_base / "index"
    already_exported: set[str] = set()
    if only_new and mapping_csv_path and Path(mapping_csv_path).exists():
        already_exported = set(pd.read_csv(mapping_csv_path)["set_id"].astype(str))

    records, subject_id_map, subjects_raw = load_and_deduplicate_entries(
        idx_path,
        existing_mapping_csv=mapping_csv_path if only_new else None,
    )

    part_csv = out_dir / "participant_summary.csv"
    part_df = _build_participant_summary(
        subject_id_map,
        subjects_raw,
        existing_csv=part_csv if only_new else None,
    )
    part_lookup = {r["participant_id"]: r for r in part_df.to_dict(orient="records")}

    sets_rows: list[dict] = []
    all_cons_dfs: list[pd.DataFrame] = []
    diag_rows: list[dict] = []

    if only_new and (out_dir / "sets_summary.csv").exists():
        sets_rows = pd.read_csv(out_dir / "sets_summary.csv").to_dict(orient="records")
    if only_new and (out_dir / "phase_markers_summary.csv").exists():
        all_cons_dfs.append(pd.read_csv(out_dir / "phase_markers_summary.csv"))

    sample_h5_written = False
    for rec in records:
        set_id = rec["_set_id"]
        if only_new and set_id in already_exported:
            continue

        raw_csv = source_base / rec["raw_csv_path"]
        raw_exists = raw_csv.exists()
        related = rec.get("related_files", {})
        front_ok = bool(related.get("front_video") and (source_base / related["front_video"]).exists())
        side_ok = bool(related.get("side_video") and (source_base / related["side_video"]).exists())

        if not raw_exists:
            print(f"⚠️ Skipping {set_id} ({rec['_source_entry_id']}): missing {raw_csv}")
            diag_rows.append(
                {
                    "set_id": set_id,
                    "participant_id": rec["_participant_id"],
                    "source_entry_id": rec["_source_entry_id"],
                    "has_faults": bool(rec["_has_faults"]),
                    "raw_csv_exists": False,
                    "imu_samples": 0,
                    "sensors_ok": 0,
                    "sensors_total": len(SENSOR_ORDER),
                    "bad_or_missing_sensors": "ALL (missing CSV)",
                    "sample_raw_cols": "",
                    "n_annotators": 0,
                    "n_markers": 0,
                    "front_video": front_ok,
                    "side_video": side_ok,
                }
            )
            continue

        imu_fs = float(rec.get("imu_fs_hz") or 100.0)
        video_fps = float(rec.get("video_fps") or 30.0)
        raw_df = pd.read_csv(raw_csv)
        imu_arrays = standardize_imu_arrays(raw_df, imu_fs_hz=imu_fs)
        n_samples = len(raw_df)
        duration_s = round(n_samples / imu_fs, 4)

        # Detailed fault summary from standardize_imu_arrays (missing, all_nan, partial_nan, flatline, freeze, clipping)
        sensor_faults = imu_arrays["sensor_faults"]
        faulty_str = "; ".join(f"{code} ({','.join(flts)})" for code, flts in sensor_faults.items()) or "NONE"
        unique_fault_types = sorted({flt for flts in sensor_faults.values() for flt in flts})
        fault_types_str = ", ".join(unique_fault_types) if unique_fault_types else "NONE"

        raw_time = str(rec.get("time", "")).strip()
        session_time = f"{raw_time}+05:30" if raw_time and "+05:30" not in raw_time else raw_time

        h5_set_row = {
            "set_id": set_id,
            "participant_id": rec["_participant_id"],
            "session_date": str(rec.get("date", "")),
            "session_time": session_time,
            "load_kg": float(rec.get("weight_kg") or 0.0),
            "lifts_total": int(rec.get("reps") or 0),
            "lifts_usable": int(rec.get("usable_reps") or 0),
            "imu_samples": n_samples,
            "duration_s": duration_s,
            "sensors_ok": int(imu_arrays["sensors_ok"]),
            "sensors_total": int(imu_arrays["sensors_total"]),
            "has_faults": bool(rec["_has_faults"]),
        }
        csv_set_row = {
            **h5_set_row,
            "source_entry_id": rec["_source_entry_id"],
            "original_experiment_id": str(rec.get("original_experiment_id", "")),
            "imu_faulty": faulty_str,
        }
        sets_rows.append(csv_set_row)

        cons_df, ann_df = _extract_phase_markers(rec, source_base, video_fps, imu_fs, n_samples)
        if not cons_df.empty:
            all_cons_dfs.append(cons_df)

        n_ann = int(ann_df["annotator_id"].nunique()) if not ann_df.empty else 0
        raw_imu_cols = [c for c in raw_df.columns if c != "Timestamp"][:3]
        diag_rows.append(
            {
                "set_id": set_id,
                "participant_id": rec["_participant_id"],
                "source_entry_id": rec["_source_entry_id"],
                "original_experiment_id": str(rec.get("original_experiment_id", "")),
                "has_faults": bool(rec["_has_faults"]),
                "raw_csv_exists": True,
                "imu_samples": n_samples,
                "sensors_ok": h5_set_row["sensors_ok"],
                "sensors_total": h5_set_row["sensors_total"],
                "fault_types_found": fault_types_str,
                "imu_faulty": faulty_str,
                "sample_raw_cols": ", ".join(raw_imu_cols),
                "n_annotators": n_ann,
                "n_markers": len(cons_df),
                "front_video": front_ok,
                "side_video": side_ok,
            }
        )

        prefix = "[DRY RUN]" if dry_run else "✓ Exported"
        print(
            f"{prefix} {set_id} ({rec['_source_entry_id']}): "
            f"sensors_ok={h5_set_row['sensors_ok']}/26, "
            f"markers={len(cons_df)} (ann={n_ann}), "
            f"videos=(F:{front_ok}, S:{side_ok}), "
            f"has_faults={h5_set_row['has_faults']}"
            + (f" | faulty={faulty_str}" if sensor_faults else "")
        )

        if dry_run and not sample_h5_written:
            sample_h5_path = Path("/tmp") / f"dry_run_{set_id}.h5"
            h5_meta = dict(h5_set_row)
            h5_meta["imu_fs_hz"] = imu_fs
            h5_part_meta = {k: v for k, v in part_lookup[rec["_participant_id"]].items() if k != "name"}
            _write_set_h5(
                sample_h5_path,
                set_meta=h5_meta,
                part_meta=h5_part_meta,
                imu_arrays=imu_arrays,
                cons_df=cons_df,
                ann_df=ann_df,
            )
            sample_h5_written = True

        if not dry_run:
            h5_meta = dict(h5_set_row)
            h5_meta["imu_fs_hz"] = imu_fs
            h5_part_meta = {k: v for k, v in part_lookup[rec["_participant_id"]].items() if k != "name"}
            _write_set_h5(
                out_dir / "sets" / f"{set_id}.h5",
                set_meta=h5_meta,
                part_meta=h5_part_meta,
                imu_arrays=imu_arrays,
                cons_df=cons_df,
                ann_df=ann_df,
            )

            if copy_videos:
                for cam_key, suffix in [("front_video", "front.mp4"), ("side_video", "side.mp4")]:
                    rel_vid = related.get(cam_key)
                    if rel_vid:
                        src_vid = source_base / rel_vid
                        if src_vid.exists():
                            shutil.copy2(src_vid, out_dir / "videos" / f"{set_id}_{suffix}")

    sets_df = pd.DataFrame(sets_rows).sort_values("set_id").reset_index(drop=True) if sets_rows else pd.DataFrame()
    cons_all_df = (
        pd.concat(all_cons_dfs, ignore_index=True)
        if all_cons_dfs
        else pd.DataFrame(columns=["set_id", "lift_id", "marker_name", "n_annotators", "mean_frame", "sd_frames", "sd_ms", "frame", "imu_idx", "time_s"])
    )
    device_df = build_device_summary_df()
    dict_df = build_data_dictionary_df()
    fault_df = pd.DataFrame(columns=FAULT_LOG_COLUMNS)
    diag_df = pd.DataFrame(diag_rows)
    mapping_df = build_id_mapping_df(records)

    if dry_run:
        full_26 = int((diag_df["sensors_ok"] == 26).sum()) if not diag_df.empty else 0
        from collections import Counter

        ft_counts: Counter[str] = Counter()
        if not diag_df.empty and "fault_types_found" in diag_df.columns:
            for val in diag_df["fault_types_found"]:
                if val and val != "NONE":
                    for ft in str(val).split(", "):
                        ft_counts[ft] += 1
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(ft_counts.items())) or "none"
        print(
            f"\n🔎 DRY RUN SUMMARY: {len(diag_df)} Sets checked | "
            f"{full_26}/{len(diag_df)} have 26/26 healthy sensors | "
            f"Participants: {len(part_df)} | Consensus markers: {len(cons_all_df)}\n"
            f"📊 Automated fault breakdown (Sets affected): {breakdown}"
        )
    else:
        part_df.to_csv(part_csv, index=False)
        sets_df.to_csv(out_dir / "sets_summary.csv", index=False)
        device_df.to_csv(out_dir / "device_summary.csv", index=False)
        dict_df.to_csv(out_dir / "data_dictionary.csv", index=False)
        cons_all_df.to_csv(out_dir / "phase_markers_summary.csv", index=False)
        if not (only_new and (out_dir / "fault_log.csv").exists()):
            fault_df.to_csv(out_dir / "fault_log.csv", index=False)

        _write_readme(out_dir)
        _write_checksums(out_dir)

        if mapping_csv_path:
            Path(mapping_csv_path).parent.mkdir(parents=True, exist_ok=True)
            mapping_df.to_csv(mapping_csv_path, index=False)

    return {
        "participants": part_df,
        "sets": sets_df,
        "devices": device_df,
        "phase_markers": cons_all_df,
        "id_mapping": mapping_df,
        "diagnostics": diag_df,
    }

