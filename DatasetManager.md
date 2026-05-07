# Wearable Biomechanics Dataset Manager

A streamlined data management system for mechatronics researchers and engineers working with wearable IMU sensors. This library automates the organization of raw experimental data (CSV) and processed outputs (Parquet), linking them to specific test subjects through a centralized JSON manifest.

## 🚀 Features

*   **Subject Management:** Track anthropometric data (height, weight, etc.) for test subjects.[cite: 2]
*   **Interactive Uploads:** Drag-and-drop CSV files directly into Google Colab; the manager renames them and saves them to Google Drive automatically.[cite: 2]
*   **Hybrid Storage:** Uses **CSV** for raw, human-readable data and **Parquet** for high-performance, compressed processed data (ideal for GNN/Kalman Filter outputs).[cite: 2]
*   **Automatic Linking:** Every processed file is tied to its parent experiment via a JSON database.[cite: 2]
*   **Data Integrity:** Includes built-in cleanup methods to delete both database entries and physical files from Drive simultaneously.[cite: 2]

---

## 📂 Directory Structure

The manager automatically creates and maintains this structure on your Google Drive:

```text
Wearable-Biomechanics/
├── db_index/
│   └── experiments.json      # Master manifest of all runs
├── subjects.json             # Registry of all participants
├── raw_data/                 # Auto-renamed original CSVs
└── processed_data/           # Subfolders organized by processing tags
    ├── kalman_filtered/      # Example Parquet files
    └── gnn_output/           # Example Parquet files
```

---

## 🛠️ Setup & Installation

Since this library is designed for Google Colab, you can pull it directly from your GitHub repository to ensure you are always using the latest version.

### 1. Import into Colab
Paste this into your first cell:

```python
import sys
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Download the manager directly from your GitHub (Example URL)
!wget -q -O biomechanics_manager.py https://raw.githubusercontent.com/janithcyapa/dataset_manager/refs/heads/main/dataset_manager.py

# Import the managers
from biomechanics_manager import SubjectManager, ExperimentManager

# Initialize
sub_db = SubjectManager()
exp_db = ExperimentManager()
```

---

## 📖 Usage Guide

### 1. Manage Test Subjects
Store participant details once and reference them in multiple experiments.[cite: 2]

```python
# Add a new person
sub_db.add_subject(
    subject_id="sub_001",
    name="John Doe",
    weight=75.5,
    height=180,
    max_lift=120,
    description="Varsity athlete, right-dominant."
)

# List all participants
sub_db.list_subjects()
```

### 2. Upload New Experimental Data
Run this after a data collection session. It will trigger an upload button.[cite: 2]

```python
exp_db.interactive_upload(
    experiment_id="exp_001",
    subject_id="sub_001",
    date="2026-05-06",
    time="14:30",
    activity_type="fast_walking",
    description="IMU on lower back. Walking on flat pavement."
)
```

### 3. Save Processed Results
After running your Kalman Filter or GNN, save the resulting DataFrame directly back to the system.[cite: 2]

```python
# Assuming 'df_results' is your processed DataFrame
exp_db.add_processed_data(
    experiment_id="exp_001",
    tag="kalman_estimates",
    processed_df=df_results
)
```

### 4. Query and Cleanup
View your research progress or delete failed runs.[cite: 2]

```python
# View all experiments and their associated file paths
exp_db.list_experiments()

# Delete an experiment (Removes the JSON entry AND deletes files from Drive)
exp_db.delete_experiment("exp_001")
```

---

## ⚙️ Requirements

*   `pandas`
*   `pyarrow` (for Parquet support)
*   `google.colab` (designed specifically for Colab environment)

---

## 📜 Research & Citations
This tool was developed as part of a research project at the **University of Peradeniya** focusing on posture estimation using Graph Neural Networks (GNN) and sensor fusion.