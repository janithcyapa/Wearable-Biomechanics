import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """\
# Joint Angle Estimation and EKF Estimation Notebook
This notebook processes raw experiment data from `OpenDatasetStore`, runs the Indirect Kalman Filter (IKF) for all available IMUs, and then estimates joint angles using the JAE logic.
"""

code_setup = """\
from google.colab import drive
import pandas as pd
import numpy as np
import os
import sys

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Install required library if needed
!pip install -q git+https://github.com/janithcyapa/OpenDatasetStore.git

from open_dataset_store import OpenDatasetStore

# Store Configuration
dataset_path = "/content/drive/MyDrive/Datasets/Wearable-Biomechanics" # Adjust as needed
store = OpenDatasetStore(
    base_dir=dataset_path,
    backend="local",
    entity_id_format="sub_{num:04d}",
    entry_id_format="{type}_{num:04d}",
    raw_filename_format="{entry_id}_{entity_id}_{ts}.csv",
    processed_filename_format="{entry_id}_{tag}.parquet"
)

# Fetch jae.py and ikf.py if running on Colab, assuming they are in the main repo
!wget -q -O ikf.py https://raw.githubusercontent.com/janithcyapa/Wearable-Biomechanics/main/ikf.py
!wget -q -O jae.py https://raw.githubusercontent.com/janithcyapa/Wearable-Biomechanics/main/jae.py

from ikf import IKF
from jae import estimate_joint_angles
from scipy.spatial.transform import Rotation as R
"""

text2 = """\
## Core Processing Function
This function takes an `experiment_id` and performs two tasks:
1. Runs IKF on all available IMUs in the raw data and saves quaternions as processed data (`tag="ikf_filtered"`).
2. Estimates joint angles for predefined joint pairings and saves them as processed data (`tag="joint_angles"`).
"""

code_func = """\
def process_experiment(exp_id):
    print(f"Processing Experiment: {exp_id}")
    # Fetch raw data
    raw_df = store.get_entry_raw_data("experiments", exp_id)
    if raw_df is None or raw_df.empty:
        print(f"No raw data found for {exp_id}")
        return
        
    timestamps = raw_df['Timestamp'].values if 'Timestamp' in raw_df.columns else np.arange(len(raw_df))
    dt = 0.01 # Assuming 100Hz, adjust if needed based on the dataset
    
    # ---------------------------------------------------------
    # Block 1: Run IKF for all devices and IMUs
    # ---------------------------------------------------------
    print("Running IKF for all IMUs...")
    ikf_results = {'Timestamp': timestamps}
    
    nodes = range(1, 11)
    imus = range(1, 4)
    
    for node in nodes:
        for imu in imus:
            # Support both column naming conventions
            acc_cols = [f'OrbiX-Node-{node}_IMU{imu}_AX', f'OrbiX-Node-{node}_IMU{imu}_AY', f'OrbiX-Node-{node}_IMU{imu}_AZ']
            gyr_cols = [f'OrbiX-Node-{node}_IMU{imu}_GX', f'OrbiX-Node-{node}_IMU{imu}_GY', f'OrbiX-Node-{node}_IMU{imu}_GZ']
            
            if not all(col in raw_df.columns for col in acc_cols + gyr_cols):
                acc_cols = [f'device{node}_IMU{imu}_AX', f'device{node}_IMU{imu}_AY', f'device{node}_IMU{imu}_AZ']
                gyr_cols = [f'device{node}_IMU{imu}_GX', f'device{node}_IMU{imu}_GY', f'device{node}_IMU{imu}_GZ']
                
            if all(col in raw_df.columns for col in acc_cols + gyr_cols):
                print(f"  Processing Node {node}, IMU {imu}...")
                acc = raw_df[acc_cols].values / 9.81  # jae/ikf expects g
                gyr = raw_df[gyr_cols].values
                
                # Convert Gyroscope to rad/s if it appears to be in degrees/s (values > 50 are a good hint)
                if np.max(np.abs(gyr)) > 50:
                    gyr = np.deg2rad(gyr)
                
                ikf_filter = IKF(dt=dt)
                matrices = ikf_filter.process(acc, gyr)
                
                # Convert rotation matrices to quaternions [x, y, z, w]
                quats = np.array([R.from_matrix(mat).as_quat() for mat in matrices])
                
                prefix = f'Node{node}_IMU{imu}'
                ikf_results[f'{prefix}_qX'] = quats[:, 0]
                ikf_results[f'{prefix}_qY'] = quats[:, 1]
                ikf_results[f'{prefix}_qZ'] = quats[:, 2]
                ikf_results[f'{prefix}_qW'] = quats[:, 3]

    if len(ikf_results.keys()) > 1:
        df_ikf = pd.DataFrame(ikf_results)
        store.add_processed_data("experiments", exp_id, tag="ikf_filtered", df=df_ikf)
        print("  -> Saved IKF processed data.")
    else:
        print("  -> No IMU data found for IKF.")
    
    # ---------------------------------------------------------
    # Block 2: Run JAE logic to estimate joint angles
    # ---------------------------------------------------------
    print("\\nRunning Joint Angle Estimation...")
    jae_results = {'Timestamp': timestamps}
    
    def get_sensor_data(node, imu):
        acc_cols_orbix = [f'OrbiX-Node-{node}_IMU{imu}_AX', f'OrbiX-Node-{node}_IMU{imu}_AY', f'OrbiX-Node-{node}_IMU{imu}_AZ']
        gyr_cols_orbix = [f'OrbiX-Node-{node}_IMU{imu}_GX', f'OrbiX-Node-{node}_IMU{imu}_GY', f'OrbiX-Node-{node}_IMU{imu}_GZ']
        acc_cols_dev = [f'device{node}_IMU{imu}_AX', f'device{node}_IMU{imu}_AY', f'device{node}_IMU{imu}_AZ']
        gyr_cols_dev = [f'device{node}_IMU{imu}_GX', f'device{node}_IMU{imu}_GY', f'device{node}_IMU{imu}_GZ']
        
        if all(col in raw_df.columns for col in acc_cols_orbix + gyr_cols_orbix):
            return raw_df[acc_cols_orbix].values / 9.81, raw_df[gyr_cols_orbix].values
        elif all(col in raw_df.columns for col in acc_cols_dev + gyr_cols_dev):
            return raw_df[acc_cols_dev].values / 9.81, raw_df[gyr_cols_dev].values
        return None, None
        
    def rotate_x(data, angle_deg):
        if angle_deg == 0 or data is None: return data
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot_m = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return np.dot(data, rot_m.T)

    all_joints = {
        "Right Knee": {"prox": (1, 2), "dist": (10, 1)},
        "Left Knee":  {"prox": (3, 2), "dist": (8, 1)},
        "Right Elbow":{"prox": (2, 2), "dist": (6, 1)},
        "Left Elbow": {"prox": (4, 2), "dist": (7, 1)},
        "Lower Back - Right Thigh (5 to 1)": {"prox": (5, 1), "dist": (1, 1), "dist_rot_x": -90},
        "Lower Back - Left Thigh (5 to 3)": {"prox": (5, 1), "dist": (3, 1), "dist_rot_x": 90},
        "Right Thigh - Left Thigh (1 to 3)": {"prox": (1, 1), "dist": (3, 1), "dist_rot_x": 180},
        "Spine": {"prox": (9, 1), "dist": (7, 1)},
        "Right Wrist": {"prox": (6, 2), "dist": (6, 3)},
        "Left Wrist": {"prox": (7, 2), "dist": (7, 3)},
    }
    
    for joint_name, nodes in all_joints.items():
        acc_prox, gyr_prox = get_sensor_data(nodes["prox"][0], nodes["prox"][1])
        acc_dist, gyr_dist = get_sensor_data(nodes["dist"][0], nodes["dist"][1])
        
        if acc_prox is not None and acc_dist is not None:
            print(f"  Estimating joint angles for: {joint_name}...")
            dist_rot_x = nodes.get("dist_rot_x", 0)
            if dist_rot_x != 0:
                acc_dist = rotate_x(acc_dist, dist_rot_x)
                gyr_dist = rotate_x(gyr_dist, dist_rot_x)
                
            if np.max(np.abs(gyr_prox)) > 50 or np.max(np.abs(gyr_dist)) > 50:
                gyr_prox = np.deg2rad(gyr_prox)
                gyr_dist = np.deg2rad(gyr_dist)
                
            result = estimate_joint_angles(
                acc_prox=acc_prox, gyr_prox=gyr_prox,
                acc_dist=acc_dist, gyr_dist=gyr_dist,
                dt=dt, n_cal_samples=100, acc_gate_threshold=0.3, euler_seq="xyz"
            )
            
            # euler_swing contains roll, pitch (flexion), yaw
            prefix = joint_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            jae_results[f'{prefix}_Roll'] = result.euler_swing[:, 0]
            jae_results[f'{prefix}_Flexion'] = result.euler_swing[:, 1]
            jae_results[f'{prefix}_Yaw'] = result.euler_swing[:, 2]

    if len(jae_results.keys()) > 1:
        df_jae = pd.DataFrame(jae_results)
        store.add_processed_data("experiments", exp_id, tag="joint_angles", df=df_jae)
        print("  -> Saved Joint Angles processed data.")
    else:
        print("  -> No joint angles to save.")
        
    print("\\nDone processing experiment!")
"""

text3 = """\
## Run Pipeline
Execute the cell below with a specific experiment ID to run both algorithms and store the processed datasets.
"""

code_run = """\
# @title Process a Specific Experiment { display-mode: "form" }
target_experiment_id = "experiments_0035" # @param {type:"string"}

process_experiment(target_experiment_id)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code_func),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code_run)
]

with open('Joint_angle_and_ekf_estimation.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
