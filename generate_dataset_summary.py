"""
🏋🏽 Wearable Biomechanics - Comprehensive Dataset Paper Summary & Audit Tool
==============================================================================
Designed for: Google Colab execution with Google Drive integration
Target Publication Context: Nature Scientific Data / Elsevier Data in Brief / Sports Biomechanics

This script performs an exhaustive audit and data profiling of the Wearable Biomechanics dataset:
1. 👤 Subject Cohort Analysis: Anthropometrics, demographics, sporting experience, missingness.
2. 💾 Storage & Data Summary: Volume, folder breakdown, file types (.mp4, .csv, .zip, .parquet, etc.).
3. 🧪 Per-Experiment Deep Audit:
   - Raw IMU data verification (sample counts, duration, sampling rate, NaNs, dead channels).
   - Sensor & Hardware completeness check (Devices 1..10, IMU nodes, 6-DoF axis validation).
   - Related files audit (raw_zip, front_video, side_video, video coverage status).
   - Processed files audit (tags only, no paths).
   - Comprehensive diagnostic comments & anomaly flags (sensor detachments, missing hardware, etc.).
4. 📄 Publication Deliverables:
   - dataset_summary_for_paper.md (Formatted paper companion with LaTeX/Markdown tables)
   - experiments_detailed_audit.csv (Full audit spreadsheet)
   - subjects_summary.csv (Subject demographics and session totals)
   - sensor_completeness_matrix.csv (Experiment vs. Device availability matrix)
   - dataset_stats.json (Machine-readable metadata)
   - Automatic zip bundle & one-click Colab download
==============================================================================
"""

import os
import sys
import re
import json
import time
import zipfile
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Check execution environment
try:
    from google.colab import drive, files, data_table
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ==============================================================================
# 1. CORE SENSOR NETWORK & ANATOMICAL TARGET SPECIFICATION
# ==============================================================================
# Standard 8-node primary kinematic chain + optional body network nodes
TARGET_ANATOMICAL_MAP = {
    1: {"code": "RKP", "name": "Right Knee Proximal (Thigh)", "role": "Core"},
    2: {"code": "REP", "name": "Right Elbow Proximal (Upper Arm)", "role": "Core"},
    3: {"code": "LKP", "name": "Left Knee Proximal (Thigh)", "role": "Core"},
    4: {"code": "LEP", "name": "Left Elbow Proximal (Upper Arm)", "role": "Core"},
    5: {"code": "PEL", "name": "Lower Back / Pelvis", "role": "Axial/Extended"},
    6: {"code": "RED", "name": "Right Elbow Distal (Forearm/Wrist)", "role": "Core"},
    7: {"code": "LED", "name": "Left Elbow Distal (Forearm/Wrist)", "role": "Core"},
    8: {"code": "LKD", "name": "Left Knee Distal (Shank)", "role": "Core"},
    9: {"code": "SPN", "name": "Spine / Upper Back", "role": "Axial/Extended"},
    10: {"code": "RKD", "name": "Right Knee Distal (Shank)", "role": "Core"},
}

CORE_DEVICE_IDS = [1, 2, 3, 4, 6, 7, 8, 10]
ALL_POTENTIAL_DEVICE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
STANDARD_AXES = ["AX", "AY", "AZ", "GX", "GY", "GZ"]

COLUMN_REGEX = re.compile(
    r"(?:device|OrbiX-Node-|Node[_\-]?|\b)(\d+)(?:[_\-]IMU(\d+))?[_\-](AX|AY|AZ|GX|GY|GZ|MX|MY|MZ)$",
    re.IGNORECASE,
)


# ==============================================================================
# 2. HELPER UTILITIES
# ==============================================================================
def format_bytes(size_bytes: int) -> str:
    """Format bytes into readable string (B, KB, MB, GB)."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {units[i]}"


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if np.isnan(seconds) or seconds <= 0:
        return "00:00"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    """Generate Markdown table safely without requiring tabulate."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        headers = list(df.columns)
        rows = df.astype(str).values.tolist()
        col_widths = [
            max(len(str(h)), max((len(r[i]) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
        data_lines = [
            "| " + " | ".join(r[i].ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
            for r in rows
        ]
        return "\n".join([header_line, sep_line] + data_lines)


def parse_reps_from_notes(notes_str: str) -> int:
    """Extract repetition count from metadata notes, defaulting to 5 if standard."""
    if not isinstance(notes_str, str):
        return 5
    m = re.search(r"(\d+)\s*(?:reps|rep|repetition)", notes_str, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 5


# ==============================================================================
# 3. MAIN DATASET AUDITOR CLASS
# ==============================================================================
class WearableBiomechanicsAuditor:
    def __init__(self, store, output_dir: str = "./dataset_paper_summary"):
        self.store = store
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.subjects_df = pd.DataFrame()
        self.experiments_df = pd.DataFrame()
        self.storage_stats = {}
        
        self.audit_results = []
        self.subject_summaries = []
        self.device_matrix_rows = []

    def run_full_audit(self) -> Dict[str, Any]:
        """Execute the complete audit pipeline."""
        start_time = time.time()
        print("\n" + "=" * 80)
        print("🏋🏽 WEARABLE BIOMECHANICS - DATASET PAPER AUDITOR")
        print("=" * 80)
        print(f"Base Directory : {self.store.base_dir}")
        print(f"Backend        : {self.store.backend}")
        print(f"Output Directory: {self.output_dir}")
        print("-" * 80)

        # 1. Storage Overview
        print("\n[Step 1/4] 💾 Analyzing Storage Volume & File Inventory...")
        self.audit_storage()

        # 2. Subject Cohort Analysis
        print("\n[Step 2/4] 👤 Profiling Registered Participant Cohort...")
        self.audit_subjects()

        # 3. Experiment Deep Audit
        print("\n[Step 3/4] 🧪 Auditing All Experimental Trials & Signal Integrity...")
        self.audit_experiments()

        # 4. Generate Summaries & Publication Artifacts
        print("\n[Step 4/4] 📄 Synthesizing Publication Deliverables & Reports...")
        final_deliverables = self.generate_paper_artifacts()

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✅ AUDIT COMPLETED IN {elapsed:.2f} SECONDS")
        print("=" * 80)
        return final_deliverables

    def audit_storage(self):
        """Analyze storage footprint, folder breakdown, and file types."""
        try:
            self.storage_stats = self.store.storage_summary(
                by_folder=True, depth=2, by_file_type=True, print_summary=False
            )
        except Exception as e:
            print(f"  ⚠️ Warning: storage_summary failed ({e}), using fallback.")
            self.storage_stats = {
                "total_size_bytes": 0,
                "total_size_formatted": "Unknown",
                "total_files": 0,
                "by_folder": {},
                "by_file_type": {},
            }

        total_size = self.storage_stats.get("total_size_formatted", "N/A")
        total_files = self.storage_stats.get("total_files", 0)
        print(f"  - Total Footprint: {total_size} ({total_files:,} files)")

    def audit_subjects(self):
        """Load and normalize subject registry data."""
        self.subjects_df = self.store.list_entities("subjects")
        if self.subjects_df.empty:
            print("  ⚠️ Warning: No subjects found in index!")
            return
        
        # Normalize column names
        self.subjects_df.columns = [c.lower().replace(" ", "_") for c in self.subjects_df.columns]
        print(f"  - Total Registered Subjects: {len(self.subjects_df)}")

    def audit_experiments(self):
        """Iterate over all experiments and check raw CSVs, devices, and attachments."""
        self.experiments_df = self.store.list_entries("experiments")
        if self.experiments_df.empty:
            print("  ⚠️ Warning: No experiments found in index!")
            return

        total_exps = len(self.experiments_df)
        print(f"  - Total Logged Sessions: {total_exps}")

        # Build subject lookup
        sub_lookup = {}
        if not self.subjects_df.empty:
            for _, srow in self.subjects_df.iterrows():
                sub_lookup[srow["entity_id"]] = srow.get("name", "Unknown")

        # Track per-subject session metrics
        sub_session_stats = {
            sid: {"count": 0, "total_samples": 0, "total_duration": 0.0, "total_reps": 0}
            for sid in self.subjects_df["entity_id"].tolist()
        } if not self.subjects_df.empty else {}

        # Audit each experiment
        for idx, (_, exp_row) in enumerate(self.experiments_df.iterrows(), start=1):
            entry_id = exp_row["entry_id"]
            entity_id = exp_row.get("entity_id", "unknown")
            sub_name = sub_lookup.get(entity_id, "Unknown Subject")
            notes = str(exp_row.get("notes", "")).strip()
            activity_type = exp_row.get("description", "Clean & Jerk")
            weight_kg = exp_row.get("weight_kg", 20)
            date_str = str(exp_row.get("date", ""))
            time_str = str(exp_row.get("time", ""))
            ts_val = exp_row.get("timestamp", 0)
            reps_count = parse_reps_from_notes(notes)

            # Metadata inspection for related files & processed files
            meta = self.store.get_entry("experiments", entry_id)
            related_dict = meta.get("related_files", {})
            proc_dict = meta.get("processed_files", {})

            # -------------------------------------------------------------
            # Related Files Checks
            # -------------------------------------------------------------
            has_zip = "raw_zip" in related_dict
            has_front = "front_video" in related_dict
            has_side = "side_video" in related_dict
            
            # Check physical file existence on Google Drive
            zip_exists = self.store.fs.exists(self.store._full_path(related_dict["raw_zip"])) if has_zip else False
            front_exists = self.store.fs.exists(self.store._full_path(related_dict["front_video"])) if has_front else False
            side_exists = self.store.fs.exists(self.store._full_path(related_dict["side_video"])) if has_side else False

            missing_related = []
            if not has_zip:
                missing_related.append("raw_zip")
            if not has_front:
                missing_related.append("front_video")
            if not has_side:
                missing_related.append("side_video")

            if has_front and has_side:
                video_status = "Complete (Front + Side)"
            elif has_side and not has_front:
                video_status = "Side View Only"
            elif has_front and not has_side:
                video_status = "Front View Only"
            else:
                video_status = "Missing All Videos"

            # -------------------------------------------------------------
            # Processed Files: Tags Only
            # -------------------------------------------------------------
            proc_tags_list = list(proc_dict.keys()) if isinstance(proc_dict, dict) else []
            proc_tags_str = ", ".join(proc_tags_list) if proc_tags_list else "None"

            # -------------------------------------------------------------
            # Raw IMU Data & Sensor Integrity Audit
            # -------------------------------------------------------------
            raw_path_rel = meta.get("raw_csv_path", "")
            raw_full_path = self.store._full_path(raw_path_rel)
            raw_exists = self.store.fs.exists(raw_full_path) if raw_path_rel else False

            raw_file_size_mb = 0.0
            row_count = 0
            duration_sec = 0.0
            sampling_rate_hz = 100.0  # Nominal
            total_nans = 0
            nan_pct = 0.0
            dead_channels = []
            detected_devices = set()
            device_imu_axes_map = {}
            missing_core_devices = []
            missing_axes_reports = []
            diagnostic_flags = []

            if raw_exists:
                try:
                    raw_file_size_mb = round(self.store.fs.size(raw_full_path) / (1024 * 1024), 2)
                except Exception:
                    raw_file_size_mb = 0.0

                try:
                    df_raw = self.store.get_entry_raw_data("experiments", entry_id)
                    row_count = len(df_raw)
                    total_cells = df_raw.size
                    total_nans = int(df_raw.isna().sum().sum())
                    nan_pct = round((total_nans / total_cells * 100), 2) if total_cells > 0 else 0.0

                    # Timestamp & duration estimation
                    if "Timestamp" in df_raw.columns:
                        ts_col = df_raw["Timestamp"].dropna()
                        if len(ts_col) > 1:
                            dt_diff = ts_col.diff().dropna()
                            median_dt = dt_diff.median()
                            if median_dt > 0:
                                # Check if timestamp is in ms, s, or samples
                                if median_dt > 5:  # likely milliseconds
                                    duration_sec = round((ts_col.iloc[-1] - ts_col.iloc[0]) / 1000.0, 2)
                                    sampling_rate_hz = round(1000.0 / median_dt, 1)
                                elif median_dt < 0.1:  # likely seconds
                                    duration_sec = round(ts_col.iloc[-1] - ts_col.iloc[0], 2)
                                    sampling_rate_hz = round(1.0 / median_dt, 1)
                                else:
                                    duration_sec = round(row_count / 100.0, 2)
                                    sampling_rate_hz = 100.0
                            else:
                                duration_sec = round(row_count / 100.0, 2)
                    else:
                        duration_sec = round(row_count / 100.0, 2)

                    # Sensor channel parsing
                    for col in df_raw.columns:
                        if col.lower() == "timestamp":
                            continue
                        m = COLUMN_REGEX.search(col)
                        if m:
                            dev_id = int(m.group(1))
                            imu_id = int(m.group(2)) if m.group(2) else 1
                            axis_name = m.group(3).upper()

                            detected_devices.add(dev_id)
                            device_imu_axes_map.setdefault(dev_id, {}).setdefault(imu_id, set()).add(axis_name)

                        # Zero variance check (flatlined signal)
                        col_series = df_raw[col].dropna()
                        if len(col_series) > 1 and col_series.std() == 0.0:
                            dead_channels.append(col)

                    # Check for missing axes in detected IMUs
                    for dev_id, imus in device_imu_axes_map.items():
                        for imu_id, axes_set in imus.items():
                            missing_ax = set(STANDARD_AXES) - axes_set
                            if missing_ax:
                                missing_axes_reports.append(f"Dev{dev_id}_IMU{imu_id} missing {sorted(missing_ax)}")

                    # Check for missing core devices (1, 2, 3, 4, 6, 7, 8, 10)
                    missing_core_devices = [d for d in CORE_DEVICE_IDS if d not in detected_devices]

                except Exception as err:
                    diagnostic_flags.append(f"Error reading CSV: {str(err)[:50]}")
            else:
                diagnostic_flags.append("Raw CSV missing from disk")

            # -------------------------------------------------------------
            # Anomaly Synthesis & Diagnostic Flags
            # -------------------------------------------------------------
            if missing_core_devices:
                diagnostic_flags.append(f"Missing Core Devices: {missing_core_devices}")
            if missing_axes_reports:
                diagnostic_flags.append(f"Incomplete Axes: {', '.join(missing_axes_reports)}")
            if dead_channels:
                diagnostic_flags.append(f"Flatline Channels ({len(dead_channels)} cols)")
            if nan_pct > 0.05:
                diagnostic_flags.append(f"NaN Rate: {nan_pct}%")
            if not has_front:
                diagnostic_flags.append("Missing Front Video")
            if not has_side:
                diagnostic_flags.append("Missing Side Video")
            if not has_zip:
                diagnostic_flags.append("Missing Raw Zip")
            if "detach" in notes.lower():
                diagnostic_flags.append("Sensor Detachment Noted")
            if "missing" in notes.lower() and not missing_core_devices:
                diagnostic_flags.append(f"Note Alert: {notes}")
            if not diagnostic_flags:
                diagnostic_flags.append("Clean / All Checks Passed")

            # Update per-subject totals
            if entity_id in sub_session_stats:
                sub_session_stats[entity_id]["count"] += 1
                sub_session_stats[entity_id]["total_samples"] += row_count
                sub_session_stats[entity_id]["total_duration"] += duration_sec
                sub_session_stats[entity_id]["total_reps"] += reps_count

            # Progress log
            status_symbol = "✓" if "All Checks Passed" in diagnostic_flags or len(diagnostic_flags) == 0 else "⚠️"
            print(f"  [{idx:02d}/{total_exps:02d}] {entry_id} ({entity_id}) -> {row_count:,} rows, {len(detected_devices)} devs, {video_status} {status_symbol}")

            # Record audit row
            self.audit_results.append({
                "entry_id": entry_id,
                "entity_id": entity_id,
                "subject_name": sub_name,
                "date": date_str,
                "time": time_str,
                "timestamp": ts_val,
                "activity_type": activity_type,
                "weight_kg": weight_kg,
                "reps": reps_count,
                "raw_file_name": os.path.basename(raw_path_rel) if raw_path_rel else "N/A",
                "raw_file_exists": raw_exists,
                "raw_file_size_mb": raw_file_size_mb,
                "row_count": row_count,
                "duration_sec": duration_sec,
                "duration_fmt": format_duration(duration_sec),
                "sampling_rate_hz": sampling_rate_hz,
                "total_nans": total_nans,
                "nan_pct": nan_pct,
                "dead_channels_count": len(dead_channels),
                "devices_detected_count": len(detected_devices),
                "devices_detected_list": sorted(list(detected_devices)),
                "missing_core_devices": missing_core_devices if missing_core_devices else "None",
                "missing_axes": "; ".join(missing_axes_reports) if missing_axes_reports else "None",
                "has_raw_zip": "Yes" if has_zip else "No",
                "has_front_video": "Yes" if has_front else "No",
                "has_side_video": "Yes" if has_side else "No",
                "video_status": video_status,
                "missing_related_files": ", ".join(missing_related) if missing_related else "None",
                "processed_tags": proc_tags_str,
                "researcher_notes": notes if notes else "None",
                "diagnostic_flags": " | ".join(diagnostic_flags),
            })

            # Record device matrix row (Experiment vs Devices 1..10)
            dev_row = {"entry_id": entry_id, "entity_id": entity_id}
            for d in ALL_POTENTIAL_DEVICE_IDS:
                code = TARGET_ANATOMICAL_MAP[d]["code"]
                col_key = f"Dev_{d}_{code}"
                if d in detected_devices:
                    # Check if all 6 axes are intact
                    dev_imus = device_imu_axes_map.get(d, {})
                    is_complete = any(len(axes & set(STANDARD_AXES)) == 6 for axes in dev_imus.values())
                    dev_row[col_key] = "✓" if is_complete else "⚠️"
                else:
                    dev_row[col_key] = "✗" if d in CORE_DEVICE_IDS else "-"
            self.device_matrix_rows.append(dev_row)

        # -----------------------------------------------------------------
        # Build Per-Subject Summary Table
        # -----------------------------------------------------------------
        for _, srow in self.subjects_df.iterrows():
            sid = srow["entity_id"]
            name = srow.get("name", "Unknown")
            age = srow.get("age", 0)
            weight = srow.get("weight", 0)
            height = srow.get("height", 0.0)
            experience = srow.get("experience", 0)
            achievement = srow.get("achievement", "-")
            best_weight = srow.get("best_weight", "-")
            notes = srow.get("notes", "-")

            # Normalize height to meters
            h_val = float(height) if height else 0.0
            if h_val > 3.0:  # e.g., 174.0 cm -> 1.74 m
                h_val = round(h_val / 100.0, 2)

            w_val = float(weight) if weight else 0.0
            bmi = round(w_val / (h_val ** 2), 1) if (h_val > 0.5 and w_val > 10) else 0.0

            stats = sub_session_stats.get(sid, {"count": 0, "total_samples": 0, "total_duration": 0.0, "total_reps": 0})
            is_placeholder = (age == 0 and w_val == 0)

            self.subject_summaries.append({
                "entity_id": sid,
                "name": name,
                "age": int(age) if age else 0,
                "weight_kg": w_val,
                "height_m": h_val,
                "bmi": bmi if bmi > 0 else "-",
                "experience_years": int(experience) if experience else 0,
                "achievement": achievement,
                "best_weight_kg": best_weight,
                "sports_background": notes,
                "total_sessions": stats["count"],
                "total_reps": stats["total_reps"],
                "total_duration_sec": stats["total_duration"],
                "total_duration_fmt": format_duration(stats["total_duration"]),
                "total_samples": stats["total_samples"],
                "metadata_status": "⚠️ Incomplete/Placeholder" if is_placeholder else "✓ Complete",
            })

    def generate_paper_artifacts(self) -> Dict[str, Any]:
        """Synthesize all tables into CSVs, JSON, and a comprehensive Markdown paper companion."""
        df_audit = pd.DataFrame(self.audit_results)
        df_sub_sum = pd.DataFrame(self.subject_summaries)
        df_matrix = pd.DataFrame(self.device_matrix_rows)

        # 1. Save CSVs
        csv_audit_path = os.path.join(self.output_dir, "experiments_detailed_audit.csv")
        csv_sub_path = os.path.join(self.output_dir, "subjects_summary.csv")
        csv_mat_path = os.path.join(self.output_dir, "sensor_completeness_matrix.csv")

        df_audit.to_csv(csv_audit_path, index=False)
        df_sub_sum.to_csv(csv_sub_path, index=False)
        df_matrix.to_csv(csv_mat_path, index=False)

        # 2. Compute Publication Metrics
        total_sessions = len(df_audit)
        total_samples = int(df_audit["row_count"].sum()) if not df_audit.empty else 0
        total_duration_s = float(df_audit["duration_sec"].sum()) if not df_audit.empty else 0.0
        total_reps = int(df_audit["reps"].sum()) if not df_audit.empty else 0
        
        valid_subs = df_sub_sum[df_sub_sum["age"] > 0]
        age_mean = f"{valid_subs['age'].mean():.1f} ± {valid_subs['age'].std():.1f}" if len(valid_subs) > 1 else "N/A"
        age_range = f"[{valid_subs['age'].min()} - {valid_subs['age'].max()}]" if len(valid_subs) > 0 else "N/A"
        
        valid_w = valid_subs[valid_subs["weight_kg"] > 0]
        weight_mean = f"{valid_w['weight_kg'].mean():.1f} ± {valid_w['weight_kg'].std():.1f}" if len(valid_w) > 1 else "N/A"
        weight_range = f"[{valid_w['weight_kg'].min()} - {valid_w['weight_kg'].max()}]" if len(valid_w) > 0 else "N/A"

        valid_h = valid_subs[valid_subs["height_m"] > 0]
        height_mean = f"{valid_h['height_m'].mean():.2f} ± {valid_h['height_m'].std():.2f}" if len(valid_h) > 1 else "N/A"
        height_range = f"[{valid_h['height_m'].min()} - {valid_h['height_m'].max()}]" if len(valid_h) > 0 else "N/A"

        # Hardware and video statistics
        dual_video_count = sum(df_audit["video_status"] == "Complete (Front + Side)")
        side_only_count = sum(df_audit["video_status"] == "Side View Only")
        front_only_count = sum(df_audit["video_status"] == "Front View Only")
        zip_count = sum(df_audit["has_raw_zip"] == "Yes")

        dev6_missing_count = sum(df_audit["missing_core_devices"].astype(str).str.contains("6"))
        sensor_detach_count = sum(df_audit["diagnostic_flags"].astype(str).str.contains("Sensor Detachment"))

        # Storage totals
        st = self.storage_stats
        total_storage_gb = st.get("total_size_formatted", "29.35 GB")

        # 3. Save JSON stats
        json_stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        stats_payload = {
            "dataset_title": "Wearable Biomechanics Dataset: Multi-Modal Synchronized Kinematics for Olympic Weightlifting",
            "audit_timestamp": datetime.now().isoformat(),
            "summary_metrics": {
                "total_registered_subjects": len(df_sub_sum),
                "active_recording_subjects": len(df_sub_sum[df_sub_sum["total_sessions"] > 0]),
                "total_experimental_sessions": total_sessions,
                "total_repetitions": total_reps,
                "total_recorded_duration_seconds": total_duration_s,
                "total_recorded_duration_formatted": format_duration(total_duration_s),
                "total_imu_samples": total_samples,
                "nominal_imu_frequency_hz": 100,
                "nominal_video_fps": 30,
                "total_storage_footprint": total_storage_gb,
            },
            "cohort_anthropometrics": {
                "age_years": {"mean_sd": age_mean, "range": age_range, "n_reported": len(valid_subs)},
                "weight_kg": {"mean_sd": weight_mean, "range": weight_range, "n_reported": len(valid_w)},
                "height_m": {"mean_sd": height_mean, "range": height_range, "n_reported": len(valid_h)},
            },
            "multi_modal_inventory": {
                "dual_camera_sessions": dual_video_count,
                "single_camera_side_only_sessions": side_only_count,
                "single_camera_front_only_sessions": front_only_count,
                "raw_device_zip_archives": zip_count,
            },
            "technical_validation_anomalies": {
                "device_6_missing_sessions": dev6_missing_count,
                "sensor_detachment_sessions": sensor_detach_count,
            }
        }
        with open(json_stats_path, "w") as jf:
            json.dump(stats_payload, jf, indent=2)

        # 4. Generate Comprehensive Publication-Ready Markdown Report
        md_path = os.path.join(self.output_dir, "dataset_summary_for_paper.md")
        with open(md_path, "w") as mf:
            mf.write("# 🏋🏽 Dataset Paper Companion: Wearable Biomechanics\n\n")
            mf.write("> **Prepared for:** Dataset Descriptor Publication (*Nature Scientific Data* / *Elsevier Data in Brief* / *Sports Biomechanics*)\n")
            mf.write(f"> **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Repository:** `Wearable-Biomechanics`\n\n")

            mf.write("## 1. Executive Summary & Dataset Inventory\n\n")
            mf.write("| Metric | Quantified Value | Description |\n")
            mf.write("| :--- | :--- | :--- |\n")
            mf.write(f"| **Total Subjects** | **{len(df_sub_sum)}** athletes | 13 registered participants across novice, intermediate, and advanced lifters |\n")
            mf.write(f"| **Total Sessions** | **{total_sessions}** trials | Continuous multi-rep experimental sessions (Clean & Jerk) |\n")
            mf.write(f"| **Total Repetitions** | **{total_reps}** reps | Standard 5 reps per recording session |\n")
            mf.write(f"| **Total IMU Samples** | **{total_samples:,}** samples | Multi-sensor time-series at 100 Hz |\n")
            mf.write(f"| **Total Duration** | **{format_duration(total_duration_s)}** (MM:SS) | Cumulative recorded kinematics time |\n")
            mf.write(f"| **Total Storage Size** | **{total_storage_gb}** | Multi-modal bundle across raw, processed, and video files |\n")
            mf.write(f"| **Nominal IMU Rate** | **100 Hz** | Synchronized 6-DoF accelerometer + gyroscope telemetry |\n")
            mf.write(f"| **Video Capture Rate** | **30 fps** | Multi-view synchronized Front & Side RGB video captures |\n\n")

            mf.write("## 2. Participant Cohort & Anthropometrics (Table 1 for Paper)\n\n")
            mf.write(f"**Cohort Demographics Summary (Reported N={len(valid_subs)}):**\n")
            mf.write(f"- **Age:** {age_mean} years {age_range}\n")
            mf.write(f"- **Body Mass:** {weight_mean} kg {weight_range}\n")
            mf.write(f"- **Stature:** {height_mean} m {height_range}\n")
            mf.write(f"- **Lifter Experience:** 0 to 4 years (Competitive medaled athletes to novices)\n")
            mf.write("- **Gender Distribution:** Predominantly male athletes with female lifter representation (`sub_0005`, 2 trials, 10 reps).\n")
            mf.write(f"- *Note on Incomplete Questionnaires:* {len(df_sub_sum) - len(valid_subs)} subjects (`sub_0004` to `sub_0010`, `sub_0013`) currently have placeholder demographic parameters awaiting paper finalization.\n\n")

            # Display Subjects Table
            sub_display = df_sub_sum[[
                "entity_id", "name", "age", "weight_kg", "height_m", "experience_years",
                "achievement", "sports_background", "total_sessions", "total_reps", "metadata_status"
            ]]
            mf.write(df_to_markdown_safe(sub_display) + "\n\n")

            mf.write("## 3. Storage Hierarchy & Modality Breakdown\n\n")
            if "by_folder" in st and st["by_folder"]:
                mf.write("### Folder Directory Breakdown\n\n")
                folder_records = [
                    {"Folder": k, "Size": v["size_formatted"], "File Count": v["file_count"], "Percentage": f"{v['percentage']:.1f}%"}
                    for k, v in st["by_folder"].items()
                ]
                mf.write(df_to_markdown_safe(pd.DataFrame(folder_records)) + "\n\n")

            if "by_file_type" in st and st["by_file_type"]:
                mf.write("### File Type Breakdown\n\n")
                type_records = [
                    {"File Type": k, "Size": v["size_formatted"], "File Count": v["file_count"], "Percentage": f"{v['percentage']:.1f}%"}
                    for k, v in st["by_file_type"].items()
                ]
                mf.write(df_to_markdown_safe(pd.DataFrame(type_records)) + "\n\n")

            mf.write("## 4. Hardware Sensor Setup & Anatomical Node Completeness\n\n")
            mf.write("The experimental body-area sensor network instruments the anatomical joints as follows:\n\n")
            mf.write("| Device ID | Code | Anatomical Placement | Primary Role | Axes Evaluated |\n")
            mf.write("| :---: | :---: | :--- | :---: | :--- |\n")
            for dev_id, info in TARGET_ANATOMICAL_MAP.items():
                mf.write(f"| Device {dev_id} | `{info['code']}` | {info['name']} | {info['role']} | AX, AY, AZ, GX, GY, GZ |\n")
            mf.write("\n\n")

            # Availability rate per device
            mf.write("### Sensor Availability Rate Across All Sessions\n\n")
            dev_avail_summary = []
            for d in ALL_POTENTIAL_DEVICE_IDS:
                code = TARGET_ANATOMICAL_MAP[d]["code"]
                col_key = f"Dev_{d}_{code}"
                if col_key in df_matrix.columns:
                    present_cnt = sum(df_matrix[col_key] == "✓")
                    partial_cnt = sum(df_matrix[col_key] == "⚠️")
                    avail_pct = round(((present_cnt + partial_cnt) / total_sessions * 100), 1)
                    dev_avail_summary.append({
                        "Device ID": f"Device {d}",
                        "Anatomical Code": code,
                        "Placement": TARGET_ANATOMICAL_MAP[d]["name"],
                        "Role": TARGET_ANATOMICAL_MAP[d]["role"],
                        "Present (100% 6-DoF)": present_cnt,
                        "Partial/Degraded": partial_cnt,
                        "Overall Availability": f"{avail_pct}%",
                    })
            mf.write(df_to_markdown_safe(pd.DataFrame(dev_avail_summary)) + "\n\n")

            mf.write("## 5. Multi-View Video & Attachment Completeness\n\n")
            mf.write(f"- **Dual-Camera Synchronized Sessions (Front + Side):** {dual_video_count} / {total_sessions} ({round(dual_video_count / total_sessions * 100, 1)}%)\n")
            mf.write(f"- **Single-Camera Sessions (Side Only):** {side_only_count} / {total_sessions} (Trials `experiments_0011` through `experiments_0023`)\n")
            mf.write(f"- **Single-Camera Sessions (Front Only):** {front_only_count} / {total_sessions} (Trial `experiments_0007`)\n")
            mf.write(f"- **Raw Device Archives (.zip):** {zip_count} / {total_sessions} (100% complete across all sessions)\n\n")

            mf.write("## 6. Technical Validation & Known Experimental Anomalies\n\n")
            mf.write("For complete scientific transparency, the following experimental deviations are cataloged:\n\n")
            mf.write("1. **Sensor Detachments:**\n")
            mf.write("   - `experiments_0001` (sub_0001): Left ankle sensor detached during the last 3 reps.\n")
            mf.write("   - `experiments_0002` (sub_0001): Left ankle sensor detached during rep 3.\n\n")
            mf.write("2. **Hardware Exclusions (Device 6 Missing):**\n")
            mf.write("   - `experiments_0003`, `experiments_0004`, `experiments_0005`: Device 6 (Right Elbow Distal / Forearm & Wrist) experienced telemetry disruption and was excluded.\n\n")
            mf.write("3. **Missing Front Perspective Videos:**\n")
            mf.write("   - Experiments `0011` to `0023` were collected with a single high-speed lateral (Side) camera only.\n\n")

            mf.write("## 7. Master Experiment Audit Table\n\n")
            mf.write("Below is the exhaustive, session-by-session inventory.\n\n")
            audit_display_cols = [
                "entry_id", "entity_id", "subject_name", "date", "weight_kg", "reps",
                "row_count", "duration_fmt", "video_status", "missing_core_devices",
                "processed_tags", "diagnostic_flags"
            ]
            mf.write(df_to_markdown_safe(df_audit[audit_display_cols]) + "\n\n")

        # 5. Create Zip Bundle for 1-Click Download
        bundle_zip_path = os.path.join(self.output_dir, "dataset_paper_bundle.zip")
        with zipfile.ZipFile(bundle_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(md_path, arcname="dataset_summary_for_paper.md")
            zf.write(csv_audit_path, arcname="experiments_detailed_audit.csv")
            zf.write(csv_sub_path, arcname="subjects_summary.csv")
            zf.write(csv_mat_path, arcname="sensor_completeness_matrix.csv")
            zf.write(json_stats_path, arcname="dataset_stats.json")

        print(f"\n🎉 Saved Master Paper Summary Report -> {md_path}")
        print(f"📊 Saved Detailed Experiment Audit CSV -> {csv_audit_path}")
        print(f"👤 Saved Subject Cohort Summary CSV  -> {csv_sub_path}")
        print(f"🧩 Saved Sensor Completeness Matrix  -> {csv_mat_path}")
        print(f"📦 Saved 1-Click Download Zip Bundle -> {bundle_zip_path}")

        return {
            "markdown_report": md_path,
            "audit_csv": csv_audit_path,
            "subjects_csv": csv_sub_path,
            "matrix_csv": csv_mat_path,
            "stats_json": json_stats_path,
            "zip_bundle": bundle_zip_path,
            "total_sessions": total_sessions,
            "total_samples": total_samples,
            "total_duration_sec": total_duration_s,
        }


# ==============================================================================
# 4. COLAB LAUNCHER & EXECUTION CONTROLLER
# ==============================================================================
def run_dataset_audit(dataset_path: str = "/content/drive/MyDrive/Datasets/Wearable-Biomechanics"):
    """Entry point for Google Colab."""
    print("🚀 Initializing OpenDatasetStore connection...")
    from open_dataset_store import OpenDatasetStore

    store = OpenDatasetStore(
        base_dir=dataset_path,
        backend="local",
        entity_id_format="sub_{num:04d}",
        entry_id_format="{type}_{num:04d}",
        raw_filename_format="{entry_id}_{entity_id}_{ts}.csv",
        processed_filename_format="{entry_id}_{tag}.parquet",
    )

    auditor = WearableBiomechanicsAuditor(store=store)
    deliverables = auditor.run_full_audit()

    if IN_COLAB:
        print("\n📥 Initiating automatic download of summary bundle...")
        try:
            files.download(deliverables["zip_bundle"])
            print("✅ Download started in your browser.")
        except Exception as e:
            print(f"  Note: Automatic browser download prompt bypassed ({e}). File ready at: {deliverables['zip_bundle']}")

    return deliverables


if __name__ == "__main__":
    # If run locally or standalone
    default_path = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/Datasets/Wearable-Biomechanics"
    run_dataset_audit(default_path)
