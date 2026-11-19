"""Canonical 26-IMU nomenclature, 12 phase markers, and raw IMU tensor standardization."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LEGACY_TO_STANDARD_MARKERS: dict[str, str] = {
    "start_rest": "start",
    "setup": "setup",
    "first_pull": "first_pull",
    "second_pull": "second_pull",
    "clean": "clean",
    "middle_rest": "intermediate_rest",
    "jerk_phase_one": "jerk_split",
    "jerk_phase_two": "jerk_recovery",
    "rest_up": "overhead_hold",
    "bring_down": "descend",
    "end_rest": "post_rest",
    "end_rep": "end",
}

STANDARD_MARKERS: list[str] = list(LEGACY_TO_STANDARD_MARKERS.values())

# Canonical 26-IMU specification (ordered by hardware node 01..10, slot 1..3)
DEVICE_SPEC: list[dict] = [
    {"device_ids": ("01",), "device_id": "01", "imu_index": 1, "short_code": "RKP", "anatomical_name": "right_knee_proximal", "side": "right", "joint": "knee", "position": "proximal"},
    {"device_ids": ("01",), "device_id": "01", "imu_index": 2, "short_code": "RHD", "anatomical_name": "right_hip_distal", "side": "right", "joint": "hip", "position": "distal"},
    {"device_ids": ("02",), "device_id": "02", "imu_index": 1, "short_code": "REP", "anatomical_name": "right_elbow_proximal", "side": "right", "joint": "elbow", "position": "proximal"},
    {"device_ids": ("02",), "device_id": "02", "imu_index": 2, "short_code": "RSD", "anatomical_name": "right_shoulder_distal", "side": "right", "joint": "shoulder", "position": "distal"},
    {"device_ids": ("03",), "device_id": "03", "imu_index": 1, "short_code": "LKP", "anatomical_name": "left_knee_proximal", "side": "left", "joint": "knee", "position": "proximal"},
    {"device_ids": ("03",), "device_id": "03", "imu_index": 2, "short_code": "LHD", "anatomical_name": "left_hip_distal", "side": "left", "joint": "hip", "position": "distal"},
    {"device_ids": ("04",), "device_id": "04", "imu_index": 1, "short_code": "LEP", "anatomical_name": "left_elbow_proximal", "side": "left", "joint": "elbow", "position": "proximal"},
    {"device_ids": ("04",), "device_id": "04", "imu_index": 2, "short_code": "LSD", "anatomical_name": "left_shoulder_distal", "side": "left", "joint": "shoulder", "position": "distal"},
    {"device_ids": ("05", "11"), "device_id": "05/11", "imu_index": 1, "short_code": "LHP", "anatomical_name": "left_hip_proximal", "side": "left", "joint": "hip", "position": "proximal"},
    {"device_ids": ("05", "11"), "device_id": "05/11", "imu_index": 2, "short_code": "RHP", "anatomical_name": "right_hip_proximal", "side": "right", "joint": "hip", "position": "proximal"},
    {"device_ids": ("05", "11"), "device_id": "05/11", "imu_index": 3, "short_code": "LS", "anatomical_name": "lower_spine", "side": "center", "joint": "spine", "position": "lower"},
    {"device_ids": ("06",), "device_id": "06", "imu_index": 1, "short_code": "RWD", "anatomical_name": "right_wrist_distal", "side": "right", "joint": "wrist", "position": "distal"},
    {"device_ids": ("06",), "device_id": "06", "imu_index": 2, "short_code": "RWP", "anatomical_name": "right_wrist_proximal", "side": "right", "joint": "wrist", "position": "proximal"},
    {"device_ids": ("06",), "device_id": "06", "imu_index": 3, "short_code": "RED", "anatomical_name": "right_elbow_distal", "side": "right", "joint": "elbow", "position": "distal"},
    {"device_ids": ("07",), "device_id": "07", "imu_index": 1, "short_code": "LWD", "anatomical_name": "left_wrist_distal", "side": "left", "joint": "wrist", "position": "distal"},
    {"device_ids": ("07",), "device_id": "07", "imu_index": 2, "short_code": "LWP", "anatomical_name": "left_wrist_proximal", "side": "left", "joint": "wrist", "position": "proximal"},
    {"device_ids": ("07",), "device_id": "07", "imu_index": 3, "short_code": "LED", "anatomical_name": "left_elbow_distal", "side": "left", "joint": "elbow", "position": "distal"},
    {"device_ids": ("08",), "device_id": "08", "imu_index": 1, "short_code": "LAD", "anatomical_name": "left_ankle_distal", "side": "left", "joint": "ankle", "position": "distal"},
    {"device_ids": ("08",), "device_id": "08", "imu_index": 2, "short_code": "LAP", "anatomical_name": "left_ankle_proximal", "side": "left", "joint": "ankle", "position": "proximal"},
    {"device_ids": ("08",), "device_id": "08", "imu_index": 3, "short_code": "LKD", "anatomical_name": "left_knee_distal", "side": "left", "joint": "knee", "position": "distal"},
    {"device_ids": ("09",), "device_id": "09", "imu_index": 1, "short_code": "LSP", "anatomical_name": "left_shoulder_proximal", "side": "left", "joint": "shoulder", "position": "proximal"},
    {"device_ids": ("09",), "device_id": "09", "imu_index": 2, "short_code": "RSP", "anatomical_name": "right_shoulder_proximal", "side": "right", "joint": "shoulder", "position": "proximal"},
    {"device_ids": ("09",), "device_id": "09", "imu_index": 3, "short_code": "US", "anatomical_name": "upper_spine", "side": "center", "joint": "spine", "position": "upper"},
    {"device_ids": ("10",), "device_id": "10", "imu_index": 1, "short_code": "RAD", "anatomical_name": "right_ankle_distal", "side": "right", "joint": "ankle", "position": "distal"},
    {"device_ids": ("10",), "device_id": "10", "imu_index": 2, "short_code": "RAP", "anatomical_name": "right_ankle_proximal", "side": "right", "joint": "ankle", "position": "proximal"},
    {"device_ids": ("10",), "device_id": "10", "imu_index": 3, "short_code": "RKD", "anatomical_name": "right_knee_distal", "side": "right", "joint": "knee", "position": "distal"},
]

SENSOR_ORDER: list[str] = [s["short_code"] for s in DEVICE_SPEC]

_RAW_COL_RE = re.compile(r"^.*?(?<!\d)(\d+)_IMU(\d+)_(AX|AY|AZ|GX|GY|GZ)$")


def build_device_summary_df(sample_rate_hz: float = 100.0) -> pd.DataFrame:
    """Return the canonical 26-IMU `device_summary.csv` DataFrame."""
    rows = []
    for spec in DEVICE_SPEC:
        rows.append(
            {
                "device_id": spec["device_id"],
                "imu_index": spec["imu_index"],
                "short_code": spec["short_code"],
                "anatomical_name": spec["anatomical_name"],
                "side": spec["side"],
                "joint": spec["joint"],
                "position": spec["position"],
                "accel_unit": "m/s^2",
                "gyro_unit": "deg/s",
                "sample_rate_hz": sample_rate_hz,
            }
        )
    return pd.DataFrame(rows)


def _index_raw_columns(columns: list[str]) -> dict[tuple[int, int, str], str]:
    lookup: dict[tuple[int, int, str], str] = {}
    for col in columns:
        m = _RAW_COL_RE.match(str(col))
        if m:
            lookup[(int(m.group(1)), int(m.group(2)), m.group(3))] = col
    return lookup


def _has_midtrial_freeze(vals: np.ndarray, min_run: int = 50) -> bool:
    """Return True if `vals` stays identically constant for `min_run` consecutive samples."""
    if len(vals) < min_run:
        return False
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return False
    # Run-length check on consecutive equal finite values
    eq = (vals[1:] == vals[:-1]) & valid[1:] & valid[:-1]
    if not np.any(eq):
        return False
    run = 1
    for flag in eq:
        if flag:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def standardize_imu_arrays(
    raw_df: pd.DataFrame,
    imu_fs_hz: float = 100.0,
    freeze_sec: float = 0.5,
) -> dict:
    """Convert raw IMU DataFrame into standardized (N, 26, 3) accel/gyro tensors.

    Coalesces device 05 and device 11 into LHP/RHP/LS, pads missing channels with NaN,
    and detects missing, all_nan, partial_nan, flatline, mid-trial freeze, and clipping.
    """
    n_samples = len(raw_df)
    n_sensors = len(DEVICE_SPEC)
    accel = np.full((n_samples, n_sensors, 3), np.nan, dtype=np.float32)
    gyro = np.full((n_samples, n_sensors, 3), np.nan, dtype=np.float32)

    col_lookup = _index_raw_columns(list(raw_df.columns))
    min_freeze_run = max(5, int(round(freeze_sec * imu_fs_hz)))
    sensor_faults: dict[str, set[str]] = {}
    sensors_ok = 0

    for s_idx, spec in enumerate(DEVICE_SPEC):
        code = spec["short_code"]
        imu_i = spec["imu_index"]
        dev_candidates = [int(d) for d in spec["device_ids"]]

        chosen_dev = None
        for dev_num in dev_candidates:
            if (dev_num, imu_i, "AX") in col_lookup or (dev_num, imu_i, "GX") in col_lookup:
                chosen_dev = dev_num
                break

        if chosen_dev is None:
            sensor_faults[code] = {"missing"}
            continue

        faults: set[str] = set()
        for target_arr, axes, clip_bound in [
            (accel, ("AX", "AY", "AZ"), 80.0),
            (gyro, ("GX", "GY", "GZ"), 2000.0),
        ]:
            for ax_idx, ax in enumerate(axes):
                col = col_lookup.get((chosen_dev, imu_i, ax))
                if col is None:
                    faults.add("missing")
                    continue
                vals = pd.to_numeric(raw_df[col], errors="coerce").to_numpy(dtype=np.float32)
                target_arr[:, s_idx, ax_idx] = vals

                nan_count = int(np.isnan(vals).sum())
                if nan_count == n_samples:
                    faults.add("all_nan")
                    continue
                if nan_count > 0:
                    faults.add("partial_nan")

                if np.nanstd(vals) == 0:
                    faults.add("flatline")
                elif _has_midtrial_freeze(vals, min_run=min_freeze_run):
                    faults.add("freeze")

                if np.nanmin(vals) < -clip_bound or np.nanmax(vals) > clip_bound:
                    faults.add("clipping")

        if faults:
            sensor_faults[code] = faults
        else:
            sensors_ok += 1

    sample_idx = np.arange(n_samples, dtype=np.int32)
    time_s = sample_idx.astype(np.float64) / float(imu_fs_hz)

    return {
        "accel": accel,
        "gyro": gyro,
        "sample_idx": sample_idx,
        "time_s": time_s,
        "sensors_ok": sensors_ok,
        "sensors_total": n_sensors,
        "sensor_faults": {k: sorted(v) for k, v in sensor_faults.items()},
    }


def build_data_dictionary_df() -> pd.DataFrame:
    """Return the complete `data_dictionary.csv` describing all published fields."""
    rows = [
        # participant_summary.csv
        ("participant_summary.csv", "participant_id", "str", "—", "Unique de-identified athlete ID (P0001, ...)"),
        ("participant_summary.csv", "age_years", "int", "years", "Participant age"),
        ("participant_summary.csv", "sex", "str", "—", "Biological sex (M / F / NA)"),
        ("participant_summary.csv", "height_m", "float", "m", "Stature in meters"),
        ("participant_summary.csv", "body_mass_kg", "float", "kg", "Body mass in kilograms"),
        ("participant_summary.csv", "bmi_kg_m2", "float", "kg/m^2", "Body mass index"),
        ("participant_summary.csv", "experience_years", "float", "years", "Weightlifting training experience"),
        ("participant_summary.csv", "achievement", "str", "—", "Highest competition / training achievement"),
        ("participant_summary.csv", "best_weight_kg", "float", "kg", "Personal best Clean & Jerk load"),
        ("participant_summary.csv", "notes", "str", "—", "Additional background notes (e.g. participation in other sports)"),
        # sets_summary.csv
        ("sets_summary.csv", "set_id", "str", "—", "Chronological Set ID (set_0001, ...)"),
        ("sets_summary.csv", "participant_id", "str", "—", "Foreign key to participant_summary.csv"),
        ("sets_summary.csv", "session_date", "str", "YYYY-MM-DD", "Date of data collection"),
        ("sets_summary.csv", "session_time", "str", "HH:MM+05:30", "Local recording time (UTC+05:30)"),
        ("sets_summary.csv", "load_kg", "float", "kg", "Barbell load for the Set"),
        ("sets_summary.csv", "lifts_total", "int", "count", "Total Clean & Jerk lifts performed in the Set"),
        ("sets_summary.csv", "lifts_usable", "int", "count", "Usable/annotated Clean & Jerk lifts in the Set"),
        ("sets_summary.csv", "imu_samples", "int", "samples", "Total 100 Hz IMU samples N"),
        ("sets_summary.csv", "duration_s", "float", "s", "Recording duration (N / imu_fs_hz)"),
        ("sets_summary.csv", "sensors_ok", "int", "count", "Number of healthy IMUs (out of 26)"),
        ("sets_summary.csv", "sensors_total", "int", "count", "Total expected IMUs (26)"),
        ("sets_summary.csv", "has_faults", "bool", "—", "True for fault-tolerant Sets; False for complete Sets"),
        # device_summary.csv
        ("device_summary.csv", "device_id", "str", "—", "Hardware node ID (01..10, 05/11 alias)"),
        ("device_summary.csv", "imu_index", "int", "—", "On-node IMU slot (1, 2, 3)"),
        ("device_summary.csv", "short_code", "str", "—", "Standardized short code (RKP..US)"),
        ("device_summary.csv", "anatomical_name", "str", "—", "Full standardized anatomical name"),
        ("device_summary.csv", "side", "str", "—", "Body side (right, left, center)"),
        ("device_summary.csv", "joint", "str", "—", "Joint or spine segment"),
        ("device_summary.csv", "position", "str", "—", "proximal, distal, upper, or lower"),
        ("device_summary.csv", "accel_unit", "str", "m/s^2", "Accelerometer physical unit"),
        ("device_summary.csv", "gyro_unit", "str", "deg/s", "Gyroscope physical unit"),
        ("device_summary.csv", "sample_rate_hz", "float", "Hz", "Nominal sampling rate (100.0)"),
        # fault_log.csv
        ("fault_log.csv", "fault_id", "str", "—", "Unique fault ID"),
        ("fault_log.csv", "set_id", "str", "—", "Foreign key to sets_summary.csv"),
        ("fault_log.csv", "sensor_code", "str", "—", "Affected IMU short code, ALL, or VIDEO"),
        ("fault_log.csv", "fault_type", "str", "—", "Fault category (flatline, dropout, clipping, nans, missing_camera, sync_issue)"),
        ("fault_log.csv", "severity", "str", "—", "minor / major / critical"),
        ("fault_log.csv", "start_imu_idx", "int", "samples", "Start sample index"),
        ("fault_log.csv", "end_imu_idx", "int", "samples", "End sample index"),
        ("fault_log.csv", "start_s", "float", "s", "Start time in seconds"),
        ("fault_log.csv", "end_s", "float", "s", "End time in seconds"),
        ("fault_log.csv", "description", "str", "—", "Detailed description of the fault"),
        # phase_markers_summary.csv
        ("phase_markers_summary.csv", "set_id", "str", "—", "Foreign key to sets_summary.csv"),
        ("phase_markers_summary.csv", "lift_id", "int", "—", "1-based lift index within the Set"),
        ("phase_markers_summary.csv", "marker_name", "str", "—", "Standardized marker (start..end)"),
        ("phase_markers_summary.csv", "n_annotators", "int", "count", "Number of annotators for this marker"),
        ("phase_markers_summary.csv", "mean_frame", "float", "frames", "Mean frame across annotators"),
        ("phase_markers_summary.csv", "sd_frames", "float", "frames", "Inter-annotator SD in frames (ddof=1)"),
        ("phase_markers_summary.csv", "sd_ms", "float", "ms", "Inter-annotator SD in milliseconds"),
        ("phase_markers_summary.csv", "frame", "int", "frames", "Consensus video frame (round(mean_frame))"),
        ("phase_markers_summary.csv", "imu_idx", "int", "samples", "Synchronized 100 Hz IMU sample index"),
        ("phase_markers_summary.csv", "time_s", "float", "s", "Synchronized timestamp in seconds"),
        # set_XXXX.h5:/imu_data
        ("set_XXXX.h5:/imu_data", "accel", "float32", "m/s^2", "Shape (N, 26, 3) accelerometer tensor"),
        ("set_XXXX.h5:/imu_data", "gyro", "float32", "deg/s", "Shape (N, 26, 3) gyroscope tensor"),
        ("set_XXXX.h5:/imu_data", "time_s", "float64", "s", "Shape (N,) timestamp vector"),
        ("set_XXXX.h5:/imu_data", "sample_idx", "int32", "samples", "Shape (N,) 0-based sample index"),
        ("set_XXXX.h5:/imu_data", "sensor_names", "str", "—", "Shape (26,) short codes matching axis 1"),
        ("set_XXXX.h5:/imu_data", "axis_names", "str", "—", "Shape (3,) ['X', 'Y', 'Z'] matching axis 2"),
    ]
    return pd.DataFrame(rows, columns=["file", "field", "dtype", "unit", "description"])

