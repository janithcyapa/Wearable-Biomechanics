# Biomechanics Framework for Injury Prediction and Rehabilitation

> Project By : Multidisciplinary AI Research Centre(MARC), University of Peradeniya

## Overview

This project implements an integrated wearable biomechanics system that unifies rigid-body mechanics with adaptive machine learning to monitor musculoskeletal health. By combining high-fidelity kinematic data from IMU sensor arrays with physics-informed neural networks, the system quantifies internal joint stresses and muscle tensions in real-time.

**Key Features:**

* **Hybrid Inference Engine:** Integrates Newtonian inverse dynamics with deep learning (LSTM) to estimate individual muscle forces and joint reaction torques.
* **Physics-Informed Modeling:** Grounded in musculoskeletal architecture and functional morphology to ensure biological plausibility in force estimations.
* **Real-Time Injury Assessment:** Continuous monitoring of biomechanical "Failure Modes," including joint-angle asymmetry (>10°), muscle-force imbalance (>15%), and peak tension thresholds (e.g., >300N for tricep strain).

**Hardware Support:**

* Designed for multi-node IMU configurations.

---

## Processing Pipeline & Notebooks

This repository includes a robust data processing pipeline consisting of three primary Jupyter notebooks. Each step is fully documented in the `Docs/` directory.

### 1. Dataset Manager (`01. Dataset Manager.ipynb`)

The dataset is managed via a central command interface bridging Google Drive and Google Colab, utilizing the `OpenDatasetStore` utility to maintain an automated JSON-indexed manifest of all experimental data.

* **Entities (Subjects):** Participants are registered with unique IDs (e.g., `sub_0001`) along with their age, weight, height, and experience.
* **Entries (Experiments):** Trials are logged as entries (e.g., `experiments_0001`) linked to specific subjects. Each experiment captures multi-modal data including raw IMU telemetry and synchronized video captures.
* **Storage Architecture:** Data is neatly organized into `index/`, `raw_data/`, `related_files/`, and `processed_data/` directories.
* **Further details:** Refer to [Docs/01. Dataset Documentation.md](<Docs/01.%20Dataset%20Documentation.md>)

### 2. Data Cleanup (`02. Data Cleanup.ipynb`)

The Data Cleaner serves as the preprocessing pipeline to extract, filter, and rename specific IMU sensor data for downstream kinematic analysis.

* **Core Processing:** Identifies valid dataset columns using dynamic regular expressions (`get_columns`) and isolates primary telemetry axes (`extract_and_rename_imus`).
* **Sensor Mapping:** Standardizes sensor node names based on hardware device IDs, producing anatomical labels like `RKP` (Right Knee Proximal), `REP` (Right Elbow Proximal), etc.
* **Workflow:** Performs data ingestion, visual QA (via Plotly), optimal IMU extraction, and finally saves the subset data as `.parquet` files tagged `selected_for_4ang`.
* **Further details:** Refer to [Docs/02. Data Cleanup.md](<Docs/02.%20Data%20Cleanup.md>)

### 3. Video Sync & 3-D Pose Estimation (`03. Video Sync.ipynb`)

This pipeline converts two unsynchronized RGB videos (front and side views) into a single, temporally coherent, root-centered 3-D skeletal sequence.

* **2-D Keypoint Extraction:** Utilizes `yolo11x-pose` to detect 17 COCO keypoints per frame in each video.
* **Temporal Synchronization:** Estimates frame lag between cameras using cross-correlation of a 1-D movement energy signal.
* **Bayesian MAP 3-D Pose Fusion:** Solves for 3-D coordinates and global bone lengths iteratively via Batch EM. The objective function minimizes weighted reprojection error while enforcing an anatomical bone-length prior and temporal smoothness.
* **Robust Optimization:** Uses `scipy.optimize.least_squares` with robust loss and sparse Jacobians to handle missing and noisy YOLO detections efficiently.
* **Further details:** Refer to [Docs/03. Video Sync &amp; 3‑D Pose Estimation.md](<Docs/03.%20Video%20Sync%20&%203%E2%80%91D%20Pose%20Estimation.md>)
