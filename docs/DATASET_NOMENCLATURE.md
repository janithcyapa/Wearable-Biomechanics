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
| 6 | `middle_rest` | `intermediate_rest` | Start of Phase 6 | **P6 — `intermediate_rest`** (`intermediate_rest` → `jerk_drive`) |
| 7 | `jerk_phase_one` | `jerk_drive` | Start of Phase 7 | **P7 — `jerk_drive`** (`jerk_drive` → `jerk_catch`) |
| 8 | `jerk_phase_two` | `jerk_catch` | Start of Phase 8 | **P8 — `jerk_catch`** (`jerk_catch` → `overhead_hold`) |
| 9 | `rest_up` | `overhead_hold` | Start of Phase 9 | **P9 — `overhead_hold`** (`overhead_hold` → `descend`) |
| 10 | `bring_down` | `descend` | Start of Phase 10 | **P10 — `descend`** (`descend` → `recovery`) |
| 11 | `end_rest` | `recovery` | Start of Phase 11 | **P11 — `recovery`** (`recovery` → `end`) |
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

### 2.2 Symmetry & Hardware Abnormality Audit

1. **Anatomical Completeness & Symmetry:** ✅ All 6 bilateral joints (`shoulder`, `elbow`, `wrist`, `hip`, `knee`, `ankle`) have complete Left/Right × Proximal/Distal coverage (24 IMUs) plus `upper_spine` and `lower_spine` (2 IMUs) = **26 IMUs**, with zero missing or duplicate anatomical slots.
2. **Paired-Device Internal Order:** ✅ Every bilateral pair (`01↔03`, `02↔04`, `06↔07`, `08↔10`) uses identical `IMU1/IMU2/IMU3` slot ordering on Left and Right, and both midline nodes (`05/11` and `09`) use `_01 = Left`, `_02 = Right`, `_03 = Spine`.
3. **Device ID Side-Numbering Flip (Quirk to Remember):** ⚠️ For `01/03`, `02/04`, and `06/07`, the **lower device ID is Right** (`01`, `02`, `06`) and the **higher device ID is Left** (`03`, `04`, `07`). For the lower leg (`08/10`), this flips: **`08` is Left** and **`10` is Right**.
4. **Hardware Replacement (`05` vs `11`):** ⚠️ Device `05` was replaced by Device `11` in later sessions (`11_IMU1/2/3` = `05_IMU1/2/3`). The export pipeline must coalesce `05` and `11` into `LHP`, `RHP`, `LS` automatically.

---

## 3. Export & Publication Pipeline Plan

### Phase A — Pre-Export Metadata & Signal Cleanup
1. **Participant Metadata Standardization:**
   - Normalize `height` units (some entries are in meters e.g. `1.74`, others in cm e.g. `178.0` → standardize to **meters** or **cm** consistently).
   - Audit missing `age`, `weight` (`body_mass`), `sex`, and `experience` fields.
2. **IMU Column Standardization (`src/export/`):**
   - Build a single canonical mapper in `src/` converting raw `{dev}_IMU{imu}_{axis}` (including `05`/`11` aliasing) to standardized anatomical names (`RKP_AX` or `right_knee_proximal_AX`).
   - Run automated sensor-health & channel-completeness checks per Set.
3. **Temporal Alignment & Lift Segmentation:**
   - Apply `video_sync_front_delta_t_*` and `video_sync_side_lag_frames` so IMU timestamps, 2D/3D pose keypoints, and consensus Lift Phase Markers share a common time/frame reference.

### Phase B — Canonical Export Structure
```text
exported_dataset/
├── README.md
├── participants.csv              # Standardized Participant metadata
├── sets.csv                      # Set-level manifest (participant_id, load_kg, fps, sync offsets, sensor completeness)
├── lifts_phases.csv              # Consensus Lift Phase Markers (12 markers / 11 phases per Lift)
└── sets/
    └── set_XXXX/
        ├── imu.parquet           # 100 Hz standardized 26-IMU telemetry
        ├── pose_2d_front.parquet # 30 Hz front keypoints
        ├── pose_2d_side.parquet  # 30 Hz side keypoints
        └── pose_3d.parquet       # 30 Hz fused 3D skeleton
```

### Phase C — Publication Reports Notebook (`05. Export & Reports.ipynb`)
- Cohort & anthropometrics summary table (mean ± SD, min–max).
- Sensor completeness matrix & signal quality summary across all Sets.
- Inter-annotator agreement (SD in frames & ms) across all 12 Phase Markers and 11 Phases.
- Exported publication-ready Plotly figures (3× high-DPI PNG/SVG) and LaTeX/Markdown tables.

