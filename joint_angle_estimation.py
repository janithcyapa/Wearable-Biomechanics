import numpy as np
import pandas as pd
import scipy.linalg
from scipy.spatial.transform import Rotation as R
import plotly.express as px
import plotly.graph_objects as go

# =====================================================================
# Lie Group Utilities (SO(3))
# =====================================================================

def hat(v):
    """Maps a 3D vector to a skew-symmetric matrix in so(3)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def exp_so3(omega):
    """Matrix exponential from so(3) to SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    K = hat(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# =====================================================================
# 1. Configuration & Mapping
# =====================================================================

# Map your links to the actual IMU column prefixes
link_mapping = {
    'Upper_Arm_Right': ['device2_IMU1', 'device2_IMU2'],
    'Lower_Arm_Right': ['device6_IMU1', 'device6_IMU2'],
    'Palm_Right': ['device3_IMU1'],
}



# =====================================================================
# 2. Automated Steady-State Detection & A-Pose
# =====================================================================

def find_steady_state(df, imu_cols, window=50):
    """
    Finds the continuous block of data where the combined variance of acceleration 
    and gyro for all IMUs is minimized across the entire dataset.
    Returns the mean acceleration and gyro readings over this block for each IMU.
    """
    steady_means = {}
    
    df_combined_var = pd.Series(0.0, index=df.index)
    
    # Calculate norms for all IMUs
    for imu in imu_cols:
        acc_cols = [f'{imu}_AX', f'{imu}_AY', f'{imu}_AZ']
        gyr_cols = [f'{imu}_GX', f'{imu}_GY', f'{imu}_GZ']
        
        acc_norm = np.linalg.norm(df[acc_cols].values, axis=1)
        gyr_norm = np.linalg.norm(df[gyr_cols].values, axis=1)
        
        # Calculate rolling variance and add to total variance score
        df_combined_var += pd.Series(acc_norm, index=df.index).rolling(window=window).var().fillna(np.inf)
        df_combined_var += pd.Series(gyr_norm, index=df.index).rolling(window=window).var().fillna(np.inf)
        
    # Find the window with the absolute lowest combined variance
    end_idx = df_combined_var.idxmin()
    start_idx = max(0, end_idx - window + 1)
    
    # Extract the means for each IMU over this exact window
    for imu in imu_cols:
        acc_cols = [f'{imu}_AX', f'{imu}_AY', f'{imu}_AZ']
        gyr_cols = [f'{imu}_GX', f'{imu}_GY', f'{imu}_GZ']
        
        acc_mean = df.loc[start_idx:end_idx, acc_cols].mean().values
        gyr_mean = df.loc[start_idx:end_idx, gyr_cols].mean().values
        
        steady_means[imu] = {
            'acc_mean': acc_mean,
            'gyr_mean': gyr_mean
        }
        
    return steady_means, start_idx, end_idx

def compute_apose_alignments(steady_means, imu_cols):
    """
    Computes R_offset for all IMUs relative to the first IMU (Master) based on the A-Pose geometry.
    """
    R_initials = {}
    
    for imu in imu_cols:
        acc_mean = steady_means[imu]['acc_mean']
        
        # Z-axis (Against gravity)
        Z = acc_mean / np.linalg.norm(acc_mean)
        
        # Assume Y-axis (Lateral) is somewhat orthogonal to Z, pointing right or left depending on anatomy.
        # For an A-pose, assuming knuckles forward, lateral is roughly along the sagittal plane.
        # We mathematically enforce Y to be exactly orthogonal to Z.
        ref_Y = np.array([0, 1.0, 0]) 
        X = np.cross(ref_Y, Z)
        
        # If the cross product is near zero (e.g. Z is pure Y), pick a different reference
        if np.linalg.norm(X) < 1e-5:
            ref_Y = np.array([1.0, 0, 0])
            X = np.cross(ref_Y, Z)
            
        X = X / np.linalg.norm(X)
        Y = np.cross(Z, X)
        
        R_initial = np.column_stack((X, Y, Z))
        R_initials[imu] = R_initial
        
    # Calculate R_offset relative to Master (IMU 1)
    master_imu = imu_cols[0]
    R_master = R_initials[master_imu]
    
    R_offsets = {}
    for imu in imu_cols:
        # Transformation from local IMU frame to Master link frame
        R_offsets[imu] = R_master.T @ R_initials[imu]
        
    return R_offsets


# =====================================================================
# 3. Multi-IMU Intrinsic EKF Step Function
# =====================================================================

def multi_imu_iekf_step(R_k, b_k_list, P_k, acc_meas_list, gyr_meas_list, R_offsets, dt, 
                        Q_static, Q_bias, R_meas_noise, alpha_Q=0.01, Q_threshold=1.0):
    """
    Performs one step of the Multi-IMU Intrinsic EKF.
    
    State:
      R_k: (3, 3) Rotation matrix of the Link
      b_k_list: List of N (3,) bias vectors
      P_k: (3 + 3N, 3 + 3N) Covariance matrix
    
    Inputs:
      acc_meas_list: List of N (3,) accelerometer measurements
      gyr_meas_list: List of N (3,) gyroscope measurements
      R_offsets: List of N (3, 3) rotation offsets mapping local IMU frame to Master link frame
    """
    N = len(acc_meas_list)
    state_dim = 3 + 3*N
    
    # -----------------------------
    # Prediction Step
    # -----------------------------
    omega_link = np.zeros(3)
    for i in range(N):
        # Correct bias and rotate into Master link frame
        omega_i_corrected = gyr_meas_list[i] - b_k_list[i]
        omega_link += R_offsets[i] @ omega_i_corrected
        
    omega_link /= N # Average angular velocity
    
    # Propagate Orientation
    R_kp1_k = R_k @ exp_so3(omega_link * dt)
    
    # Biases remain constant
    b_kp1_k_list = [np.copy(b) for b in b_k_list]
    
    # Dynamic Process Noise (Q Tuning)
    omega_norm = np.linalg.norm(omega_link)
    # Inflate Q if moving fast, deflate if static (keeps it locked during lockout phase)
    Q_R_dynamic = Q_static + alpha_Q * max(0, omega_norm - Q_threshold) * np.eye(3)
    
    Q = np.zeros((state_dim, state_dim))
    Q[0:3, 0:3] = Q_R_dynamic
    for i in range(N):
        Q[3+3*i : 6+3*i, 3+3*i : 6+3*i] = Q_bias
        
    # State Transition Matrix Phi
    Phi = np.eye(state_dim)
    # In Lie algebra, Adjoint governs error propagation. Simplified Phi for bias:
    # dR = dR - dt * R * db_i
    for i in range(N):
        Phi[0:3, 3+3*i : 6+3*i] = -dt * R_offsets[i] / N
        
    # Predict Covariance
    P_kp1_k = Phi @ P_k @ Phi.T + Q
    
    # -----------------------------
    # Measurement Update
    # -----------------------------
    gravity = np.array([0, 0, 1.0]) # Expected gravity in inertial frame
    
    # Stack measurements
    Z_meas = np.zeros(3*N)
    Z_hat = np.zeros(3*N)
    H = np.zeros((3*N, state_dim))
    
    for i in range(N):
        # Rotate local accel measurement to Master link frame
        acc_master = R_offsets[i] @ acc_meas_list[i]
        
        # Normalize (assuming measurement is pure gravity direction)
        acc_norm = np.linalg.norm(acc_master)
        if acc_norm > 1e-5:
            acc_master /= acc_norm
            
        Z_meas[3*i : 3*i+3] = acc_master
        
        # Predicted measurement in body frame
        z_hat_i = R_kp1_k.T @ gravity
        Z_hat[3*i : 3*i+3] = z_hat_i
        
        # Observation Jacobian (d(R^T g) / d(theta) = hat(R^T g))
        H[3*i : 3*i+3, 0:3] = hat(z_hat_i)
        # H wrt biases is zero
        
    # Innovation
    innov = Z_meas - Z_hat
    
    # Measurement Noise
    R_mat = np.eye(3*N) * R_meas_noise
    
    # Kalman Gain
    S = H @ P_kp1_k @ H.T + R_mat
    K = P_kp1_k @ H.T @ np.linalg.inv(S)
    
    # State Correction
    delta_x = K @ innov
    
    delta_theta = delta_x[0:3]
    R_kp1_kp1 = R_kp1_k @ exp_so3(delta_theta)
    
    b_kp1_kp1_list = []
    for i in range(N):
        delta_b_i = delta_x[3+3*i : 6+3*i]
        b_kp1_kp1_list.append(b_kp1_k_list[i] + delta_b_i)
        
    # Update Covariance
    I_KH = np.eye(state_dim) - K @ H
    P_kp1_kp1 = I_KH @ P_kp1_k @ I_KH.T + K @ R_mat @ K.T
    
    return R_kp1_kp1, b_kp1_kp1_list, P_kp1_kp1


# =====================================================================
# 4. Main Processing Loop
# =====================================================================

def process_dataset(df, dt=0.01):
    """
    Processes the entire dataframe through the Multi-IMU EKF to extract Link orientations.
    """
    link_orientations = {link: [] for link in link_mapping.keys()}
    
    # EKF Tuning Parameters
    Q_static = 1e-4 * np.eye(3)      # Baseline gyro process noise
    Q_bias = 1e-7 * np.eye(3)        # Bias random walk
    R_meas_noise = 0.05              # Accel measurement noise variance
    
    # Filter the df to only process rows that have the necessary columns
    # We assume 'df' has been loaded and cleaned prior to this.
    
    for link, imus in link_mapping.items():
        print(f"Processing Link: {link} with IMUs {imus}")
        
        # Check if all IMUs for this link exist in the DataFrame
        missing_imus = [imu for imu in imus if f'{imu}_AX' not in df.columns]
        if missing_imus:
            print(f"  -> Skipping {link} because columns for {missing_imus} were not found.")
            continue
            
        N = len(imus)
        
        # 1. Automated Steady State & Alignment
        try:
            steady_means, _, _ = find_steady_state(df, imus)
        except ValueError as e:
            print(f"  -> Skipping {link} due to error: {e}")
            continue
            
        R_offsets_dict = compute_apose_alignments(steady_means, imus)
        R_offsets = [R_offsets_dict[imu] for imu in imus]
        
        # 2. Initialize EKF State
        R_k = np.eye(3) 
        b_k_list = [steady_means[imu]['gyr_mean'] for imu in imus] # Initialize bias from steady state
        
        state_dim = 3 + 3*N
        P_k = np.eye(state_dim) * 0.1
        
        link_rotations = []
        
        # 3. Time Loop
        for index, row in df.iterrows():
            acc_meas_list = []
            gyr_meas_list = []
            
            for imu in imus:
                acc_cols = [f'{imu}_AX', f'{imu}_AY', f'{imu}_AZ']
                gyr_cols = [f'{imu}_GX', f'{imu}_GY', f'{imu}_GZ']
                
                acc_meas_list.append(row[acc_cols].values.astype(float))
                gyr_meas_list.append(row[gyr_cols].values.astype(float))
                
            # EKF Step
            R_k, b_k_list, P_k = multi_imu_iekf_step(
                R_k, b_k_list, P_k, 
                acc_meas_list, gyr_meas_list, R_offsets, 
                dt, Q_static, Q_bias, R_meas_noise
            )
            
            # Store orientation as quaternion (w, x, y, z)
            r = R.from_matrix(R_k)
            quat = r.as_quat() # SciPy format: [x, y, z, w]
            link_rotations.append([quat[3], quat[0], quat[1], quat[2]]) # reorder to [w, x, y, z]
            
        link_orientations[link] = link_rotations
        print(f"  -> Successfully processed {len(link_rotations)} frames for {link}.")
        
    return link_orientations


# =====================================================================
# Execution & Example Plotting (from user request)
# =====================================================================
if __name__ == "__main__":
    # Load the specific dataset
    df_path = "/home/jazz/Projects/Wearable-Biomechanics/experiments_0001_sub_0001_1778229520.csv"
    try:
        df = pd.read_csv(df_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Your example Plotly figure
        # if all(col in df.columns for col in ['device1_IMU1_AX','device1_IMU1_AY','device1_IMU1_AZ', 'Timestamp']):
        #     fig = px.line(
        #         df, 
        #         x='Timestamp', 
        #         y=['device1_IMU1_AX','device1_IMU1_AY','device1_IMU1_AZ'], 
        #         title='Device 1 IMU Accelerometer Data Over Time'
        #     )
        #     # fig.show()  # Uncomment this to render the plot in your IDE/Notebook
        # else:
        #     print("Columns for Plotly example were not found in the dataset.")
            
        # Run the full pipeline
        link_orientations = process_dataset(df, dt=0.01) # Set your actual dt here
        # Example to access data: df_upper_arm_right = pd.DataFrame(link_orientations['Upper_Arm_Right'], columns=['qw', 'qx', 'qy', 'qz'])
        
    except FileNotFoundError:
        print(f"Could not find dataset at: {df_path}")
