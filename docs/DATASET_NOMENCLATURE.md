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
/sets_summary.csv               ← One row per Set (set_id, participant_id, session_date, load_kg, lifts, usable_lifts, tier, view_mode, sync offsets)
/participant_summary.csv        ← One row per Participant (de-identified: participant_id, age, sex, height_m, body_mass_kg, experience_years, achievement, best_weight_kg)
/device_summary.csv             ← Sensor ID / short code → full anatomical name, segment, side, position, axes, units
/fault_log.csv                  ← One row per sensor fault (set_id, sensor_code, fault_type, start_idx, end_idx, start_s, end_s)
/phase_markers_summary.csv      ← Flat table: set_id, lift_id, phase_marker, annotator_id, frame, imu_idx, time_s
/sets/
    └── set_XXXX.h5             ← Per-Set HDF5 archive (gzip level 4)
/videos/
    ├── set_XXXX_front.mp4      ← 30 fps H.264 front camera
    ├── set_XXXX_side.mp4       ← 30 fps H.264 side camera
    └── set_XXXX_side_by_side.mp4 ← Temporally synced side-by-side H.264 video
/checksums.sha256               ← SHA-256 verification manifest for all files
```

### 3.2 HDF5 Layout (`/sets/set_XXXX.h5`)

```text
/metadata                       attrs: set_id, participant_id, session_date, load_kg, lifts, usable_lifts, tier, view_mode, video_fps, imu_fs_hz, front_to_imu_lag_samples, side_to_front_lag_frames
/participant_metadata           attrs: participant_id, age, sex, height_m, body_mass_kg, experience_years
/imu/
    ├── accel                   (N, 26, 3) float32   attrs: units="m/s^2", sample_rate_hz=100.0
    ├── gyro                    (N, 26, 3) float32   attrs: units="deg/s", sample_rate_hz=100.0
    ├── time_s                  (N,)       float64   attrs: units="s"
    ├── sensor_names            (26,)      S4        ["RKP", "RHD", ..., "US"]
    └── axis_names              (3,)       S2        ["X", "Y", "Z"]
/pose/                          (optional, if available for the Set)
    ├── pose_2d_front           (F_f, 17, 3) float32 attrs: coords=["x","y","conf"], fps=30.0
    ├── pose_2d_side            (F_s, 17, 3) float32 attrs: coords=["x","y","conf"], fps=30.0
    └── pose_3d                 (F_3, 17, 3) float32 attrs: units="m", fps=30.0
/phase_markers/
    ├── lift_id                 (M,)       int32
    ├── marker_name             (M,)       S20       ["start", "setup", ..., "end"]
    ├── annotator_id            (M,)       S16       e.g. "consensus" or "a2"
    ├── frame                   (M,)       int32
    ├── imu_idx                 (M,)       int32
    └── time_s                  (M,)       float64
```

### 3.3 File Format Standards

| Artifact | Format | Notes |
| --- | --- | --- |
| Summary & dictionary tables | **CSV (UTF-8, no index)** | Journal-standard, human-readable, compatible with Python/R/Excel. |
| Per-Set multi-modal data | **HDF5 (`.h5`, `gzip` level 4)** | Separates `accel` (`m/s^2`) and `gyro` (`deg/s`), stores `pose` and `phase_markers` with pre-computed `imu_idx` and `time_s`. |
| Video files | **MP4 (`libx264`, `yuv420p`)** | Universal playback and OpenCV/FFmpeg compatibility. |
| Integrity manifest | **SHA-256 (`checksums.sha256`)** | Verifiable via `sha256sum -c checksums.sha256`. |
