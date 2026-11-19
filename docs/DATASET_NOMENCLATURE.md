# Clean & Jerk Wearable Biomechanics — Dataset Nomenclature & Export Plan

## 1. Working Definitions & Data Hierarchy

| Term | Definition | Current Store Mapping |
| --- | --- | --- |
| **Participant** | An athlete who performs Clean & Jerk lifts for data collection. | `subjects` entity (`sub_XXXX`) |
| **Session** | A single day of data collection (may contain multiple sets; optional metadata). | Date metadata on entries |
| **Set** | A single continuous recording containing **3–5 consecutive Clean & Jerk lifts** by the same participant. | `experiments_XXXX` / `complete_trials_XXXX` / `fault_tolerant_trials_XXXX` |
| **Lift** | A single Clean & Jerk repetition within a Set. | `rep` (`1..N`) |
| **Phases** | The **11** temporal movement segments of a Lift. | Intervals between consecutive Phase Markers |
| **Phase Markers** | The **12** timestamps / frame indices separating the 11 phases. Each marker denotes the **start** of its phase, except `end_rep` which marks the **end** of the final phase (`end_rest`) and the Lift. | `rep_segments_a{id}` columns |
| **Metadata** | Participant anthropometrics and background (`age`, `sex`, `height`, `body_mass`, `training_experience`, `achievement`, `best_weight`). | `subjects` & `experiments` manifests |

### 1.1 The 11 Phases & 12 Standardized Phase Markers (Max 2 Words)

| # | Legacy Annotation Tag | Standardized Marker | Role | Phase # & Name (`[Marker_i, Marker_{i+1})`) |
| --- | --- | --- | --- | --- |
| 1 | `start_rest` | `start` | Start of Phase 1 | **P1 — `pre_rest`** (`start` → `setup`) |
| 2 | `setup` | `setup` | Start of Phase 2 | **P2 — `setup`** (`setup` → `first_pull`) |
| 3 | `first_pull` | `first_pull` | Start of Phase 3 | **P3 — `first_pull`** (`first_pull` → `second_pull`) |
| 4 | `second_pull` | `second_pull` | Start of Phase 4 | **P4 — `second_pull`** (`second_pull` → `clean`) |
| 5 | `clean` | `clean` | Start of Phase 5 | **P5 — `clean`** (`clean` → `intermediate_rest`) |
| 6 | `middle_rest` | `intermediate_rest` | Start of Phase 6 | **P6 — `intermediate_rest`** (`intermediate_rest` → `jerk_split`) |
| 7 | `jerk_phase_one` | `jerk_split` | Start of Phase 7 | **P7 — `jerk_split`** (`jerk_split` → `jerk_recovery`) |
| 8 | `jerk_phase_two` | `jerk_recovery` | Start of Phase 8 | **P8 — `jerk_recovery`** (`jerk_recovery` → `overhead_hold`) |
| 9 | `rest_up` | `overhead_hold` | Start of Phase 9 | **P9 — `overhead_hold`** (`overhead_hold` → `descend`) |
| 10 | `bring_down` | `descend` | Start of Phase 10 | **P10 — `descend`** (`descend` → `post_rest`) |
| 11 | `end_rest` | `post_rest` | Start of Phase 11 | **P11 — `post_rest`** (`post_rest` → `end`) |
| 12 | `end_rep` | `end` | End of Phase 11 & Lift | — |

---

## 2. Standardized 26-IMU Anatomical Nomenclature

All **26 IMUs** (mounted across **10 multi-IMU sensor nodes**) follow a unified anatomical naming scheme:

- **Limb / Joint IMUs (24):** `{side}_{joint}_{position}`
  - `side`: `left` (`L`) or `right` (`R`)
  - `joint`: `shoulder` (`S`), `elbow` (`E`), `wrist` (`W`), `hip` (`H`), `knee` (`K`), `ankle` (`A`)
  - `position`: `proximal` (`P`) or `distal` (`D`)
- **Spine IMUs (2):** `upper_spine` (`US`), `lower_spine` (`LS`)
- **Channel suffix:** `{short_code}_{axis}` or `{full_name}_{axis}` where `axis` ∈ `{AX, AY, AZ, GX, GY, GZ}` (with optional `raw` prefix normalized during export).

### 2.1 Complete Device-to-Anatomical Mapping

| Device_IMU | Short Code | Full Standardized Name | Side | Segment / Joint | Position |
| --- | --- | --- | --- | --- | --- |
| `01_01` | `RKP` | `right_knee_proximal` | Right | Knee | Proximal |
| `01_02` | `RHD` | `right_hip_distal` | Right | Hip | Distal |
| `02_01` | `REP` | `right_elbow_proximal` | Right | Elbow | Proximal |
| `02_02` | `RSD` | `right_shoulder_distal` | Right | Shoulder | Distal |
| `03_01` | `LKP` | `left_knee_proximal` | Left | Knee | Proximal |
| `03_02` | `LHD` | `left_hip_distal` | Left | Hip | Distal |
| `04_01` | `LEP` | `left_elbow_proximal` | Left | Elbow | Proximal |
| `04_02` | `LSD` | `left_shoulder_distal` | Left | Shoulder | Distal |
| `05_01` / `11_01` | `LHP` | `left_hip_proximal` | Left | Hip | Proximal |
| `05_02` / `11_02` | `RHP` | `right_hip_proximal` | Right | Hip | Proximal |
| `05_03` / `11_03` | `LS` | `lower_spine` | Center | Spine | Lower |
| `06_01` | `RWD` | `right_wrist_distal` | Right | Wrist | Distal |
| `06_02` | `RWP` | `right_wrist_proximal` | Right | Wrist | Proximal |
| `06_03` | `RED` | `right_elbow_distal` | Right | Elbow | Distal |
| `07_01` | `LWD` | `left_wrist_distal` | Left | Wrist | Distal |
| `07_02` | `LWP` | `left_wrist_proximal` | Left | Wrist | Proximal |
| `07_03` | `LED` | `left_elbow_distal` | Left | Elbow | Distal |
| `08_01` | `LAD` | `left_ankle_distal` | Left | Ankle | Distal |
| `08_02` | `LAP` | `left_ankle_proximal` | Left | Ankle | Proximal |
| `08_03` | `LKD` | `left_knee_distal` | Left | Knee | Distal |
| `09_01` | `LSP` | `left_shoulder_proximal` | Left | Shoulder | Proximal |
| `09_02` | `RSP` | `right_shoulder_proximal` | Right | Shoulder | Proximal |
| `09_03` | `US` | `upper_spine` | Center | Spine | Upper |
| `10_01` | `RAD` | `right_ankle_distal` | Right | Ankle | Distal |
| `10_02` | `RAP` | `right_ankle_proximal` | Right | Ankle | Proximal |
| `10_03` | `RKD` | `right_knee_distal` | Right | Knee | Distal |


---


## 3. Published Dataset Folder Structure & File Formats

### 3.1 Top-Level Directory Layout

```text
/README.md                      ← How to start, licence, citation
/data_dictionary.csv            ← Every field, unit, type, description
/sets_summary.csv               ← One row per Set 
/participant_summary.csv        ← One row per Participant
/device_summary.csv             ← Sensor IDs and naming, units and specs
/fault_log.csv                  ← One row per sensor fault
/phase_markers_summary.csv      ← Flat table
/sets/
    └── set_XXXX.h5             ← Per-Set HDF5 archive (gzip level 4)
/videos/
    ├── set_XXXX_front.mp4      ← 30 fps H.264 front camera
    ├── set_XXXX_side.mp4       ← 30 fps H.264 side camera
/checksums.sha256               ← SHA-256 verification manifest for all files
```

### 3.2 HDF5 Layout (`/sets/set_XXXX.h5`)

```text
/metadata                       attrs
/participant_metadata           attrs of participant
/imu_data
/phase_markers
```

### 3.3 File Format Standards

| Artifact | Format | Notes |
| --- | --- | --- |
| Summary & dictionary tables | **CSV (UTF-8, no index)** | Journal-standard, human-readable, compatible with Python/R/Excel. |
| Per-Set multi-modal data | **HDF5 (`.h5`, `gzip` level 4)** | Bundles `/metadata`, `/participant_metadata`, `/imu_data`, and `/phase_markers` with pre-computed `imu_idx` and `time_s`. |
| Video files | **MP4 (`libx264`, `yuv420p`)** | Universal playback and OpenCV/FFmpeg compatibility. |
| Integrity manifest | **SHA-256 (`checksums.sha256`)** | Verifiable via `sha256sum -c checksums.sha256`. |


---

## 4. Exact Schema of Every Published File

### 4.1 `/data_dictionary.csv`
| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `file` | `str` | — | File path or group (`participant_summary.csv`, `set_XXXX.h5:/imu_data`, etc.) |
| `field` | `str` | — | Column, dataset, or attribute name |
| `dtype` | `str` | — | Data type (`str`, `int32`, `float32`, `float64`, `bool`) |
| `unit` | `str` | — | Physical unit (`m/s^2`, `deg/s`, `kg`, `m`, `s`, `Hz`, `frames`, `years`, or `—`) |
| `description` | `str` | — | Human-readable definition |

### 4.2 `/participant_summary.csv` (De-identified)
| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `participant_id` | `str` | — | Unique de-identified athlete ID (`sub_0001`, …) |
| `age_years` | `int` | `years` | Participant age (nullable if unrecorded) |
| `sex` | `str` | — | Biological sex (`M` / `F` / `NA`) |
| `height_m` | `float` | `m` | Stature normalized to meters |
| `body_mass_kg` | `float` | `kg` | Body mass |
| `bmi_kg_m2` | `float` | `kg/m^2` | Body mass index (`body_mass_kg / height_m^2`) |
| `experience_years` | `float` | `years` | Weightlifting training experience |
| `achievement` | `str` | — | Best competition / training level |
| `best_weight_kg` | `float` | `kg` | Personal best Clean & Jerk load |

### 4.3 `/sets_summary.csv`
| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `set_id` | `str` | — | Unique Set ID (`set_0001`, …) |
| `participant_id` | `str` | — | Foreign key to `participant_summary.csv` |
| `session_date` | `str` | `YYYY-MM-DD` | Date of data collection |
| `load_kg` | `float` | `kg` | Barbell load for the Set |
| `lifts_total` | `int` | `count` | Total Clean & Jerk lifts performed in the Set |
| `lifts_usable` | `int` | `count` | Number of usable/annotated lifts |
| `tier` | `str` | — | Quality tier (`complete` / `fault_tolerant`) |
| `view_mode` | `str` | — | Camera coverage (`dual_view` / `front_only` / `side_only`) |
| `video_fps` | `float` | `Hz` | Video frame rate (e.g. `30.0`) |
| `imu_fs_hz` | `float` | `Hz` | IMU sampling rate (`100.0`) |
| `imu_samples` | `int` | `samples` | Total IMU samples `N` |
| `duration_s` | `float` | `s` | Recording duration (`N / imu_fs_hz`) |
| `front_to_imu_lag_samples` | `int` | `samples` | `t_front - t_imu` in 100 Hz samples |
| `front_to_imu_lag_sec` | `float` | `s` | `t_front - t_imu` in seconds |
| `side_to_front_lag_frames` | `int` | `frames` | Frame offset between front and side cameras |
| `sensors_ok` | `int` | `count` | Number of healthy IMUs (out of `26`) |
| `sensors_total` | `int` | `count` | Total expected IMUs (`26`) |
| `has_faults` | `bool` | — | `True` if any entry exists in `fault_log.csv` |

### 4.4 `/device_summary.csv`
| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `device_id` | `str` | — | Hardware node ID (`01`..`10`, with `05/11` alias) |
| `imu_index` | `int` | — | On-node IMU slot (`1`, `2`, `3`) |
| `short_code` | `str` | — | Standardized 2–3 letter code (`RKP`, …, `US`) |
| `anatomical_name` | `str` | — | Full standardized name (`right_knee_proximal`, …) |
| `side` | `str` | — | `right`, `left`, or `center` |
| `joint` | `str` | — | `shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle`, or `spine` |
| `position` | `str` | — | `proximal`, `distal`, `upper`, or `lower` |
| `accel_unit` | `str` | — | `m/s^2` |
| `gyro_unit` | `str` | — | `deg/s` |
| `sample_rate_hz` | `float` | `Hz` | `100.0` |

### 4.5 `/fault_log.csv`
| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `fault_id` | `str` | — | Unique fault identifier (`fault_0001`, …) |
| `set_id` | `str` | — | Foreign key to `sets_summary.csv` |
| `sensor_code` | `str` | — | Affected IMU short code (`RKP`, …) or `ALL` / `VIDEO` |
| `fault_type` | `str` | — | Category (`flatline`, `dropout`, `clipping`, `nans`, `missing_camera`, `sync_issue`) |
| `severity` | `str` | — | `minor` / `major` / `critical` |
| `start_imu_idx` | `int` | `samples` | Start sample index (`0` if whole Set) |
| `end_imu_idx` | `int` | `samples` | End sample index (`N-1` if whole Set) |
| `start_s` | `float` | `s` | Start time in seconds |
| `end_s` | `float` | `s` | End time in seconds |
| `description` | `str` | — | Specific free-text description of the fault |

### 4.6 `/phase_markers_summary.csv`
| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `set_id` | `str` | — | Foreign key to `sets_summary.csv` |
| `lift_id` | `int` | — | 1-based lift index within the Set (`1..lifts_total`) |
| `marker_name` | `str` | — | One of the 12 standardized markers (`start`..`end`) |
| `annotator_id` | `str` | — | `consensus` or individual annotator ID (`a1`..`a15`) |
| `frame` | `int` | `frames` | Video frame index (at `video_fps`) |
| `imu_idx` | `int` | `samples` | Synchronized 100 Hz IMU sample index |
| `time_s` | `float` | `s` | Synchronized timestamp (`imu_idx / imu_fs_hz`) |

### 4.7 `/sets/set_XXXX.h5` (Per-Set HDF5 Schema)
- **`/metadata` (Group Attributes):** `set_id`, `participant_id`, `session_date`, `load_kg`, `lifts_total`, `lifts_usable`, `tier`, `view_mode`, `video_fps`, `imu_fs_hz`, `front_to_imu_lag_samples`, `side_to_front_lag_frames`
- **`/participant_metadata` (Group Attributes):** `participant_id`, `age_years`, `sex`, `height_m`, `body_mass_kg`, `experience_years`
- **`/imu_data/` (Group):**
  - `accel`: `(N, 26, 3)` `float32`, `units="m/s^2"`, `sample_rate_hz=100.0`
  - `gyro`: `(N, 26, 3)` `float32`, `units="deg/s"`, `sample_rate_hz=100.0`
  - `time_s`: `(N,)` `float64`, `units="s"`
  - `sample_idx`: `(N,)` `int32`
  - `sensor_names`: `(26,)` `S4` (`["RKP", "RHD", ..., "US"]` in canonical order)
  - `axis_names`: `(3,)` `S2` (`["X", "Y", "Z"]`)
- **`/phase_markers/` (Group):**
  - `lift_id`: `(M,)` `int32`
  - `marker_name`: `(M,)` `S20`
  - `annotator_id`: `(M,)` `S16`
  - `frame`: `(M,)` `int32`
  - `imu_idx`: `(M,)` `int32`
  - `time_s`: `(M,)` `float64`

### 4.8 `/videos/` & `/checksums.sha256`
- **`/videos/set_XXXX_front.mp4` & `set_XXXX_side.mp4`**: H.264 (`libx264`), `yuv420p`, `30.0 fps` (matching `video_fps` in `sets_summary.csv`).
- **`/checksums.sha256`**: Standard `sha256sum` output covering all CSV, `.h5`, `.mp4`, and `README.md` files.
