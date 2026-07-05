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
# link_mapping = {
#     'Upper_Arm_Right': ['device2_IMU1'],
#     'Lower_Arm_Right': ['device6_IMU1'],
#     'Palm_Right': ['device6_IMU3'],
# }

link_mapping = {
    'Upper_Arm_Right': ['device2_IMU1', 'device2_IMU2'],
    'Lower_Arm_Right': ['device6_IMU1', 'device6_IMU2'],
    'Palm_Right': ['device6_IMU3'],
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
        
    return R_offsets, R_master


# =====================================================================
# 3. Multi-IMU Intrinsic EKF Step Function
# =====================================================================

def multi_imu_iekf_step(R_k, b_k_list, P_k, acc_meas_list, gyr_meas_list, R_offsets, dt, 
                        Q_static, Q_bias, R_meas_noise, alpha_Q=0.01, Q_threshold=0.5):
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
    
    # Measurement Noise Matrix
    R_mat = np.zeros((3*N, 3*N))
    
    for i in range(N):
        acc_raw = acc_meas_list[i]
        acc_raw_norm = np.linalg.norm(acc_raw)
        
        # Rotate local accel measurement to Master link frame
        acc_master = R_offsets[i] @ acc_raw
        
        # Normalize (assuming measurement is pure gravity direction)
        if acc_raw_norm > 1e-5:
            acc_master /= acc_raw_norm
            
        Z_meas[3*i : 3*i+3] = acc_master
        
        # Predicted measurement in body frame
        z_hat_i = R_kp1_k.T @ gravity
        Z_hat[3*i : 3*i+3] = z_hat_i
        
        # Observation Jacobian (d(R^T g) / d(theta) = hat(R^T g))
        H[3*i : 3*i+3, 0:3] = hat(z_hat_i)
        
        # Motion Artifact Rejection (Gating)
        g_local = 9.81
        motion_penalty = (acc_raw_norm - g_local)**2
        k_penalty = 10.0 # Tune this to punish deviations from 1g
        R_meas_noise_i = R_meas_noise * (1 + k_penalty * motion_penalty)
        R_mat[3*i : 3*i+3, 3*i : 3*i+3] = np.eye(3) * R_meas_noise_i
        
    # Innovation
    innov = Z_meas - Z_hat
    
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
    
    # Find a single GLOBAL steady state for all used IMUs to ensure a shared reference frame
    all_imus = [imu for imus in link_mapping.values() for imu in imus]
    try:
        global_steady_means, start_idx, end_idx = find_steady_state(df, all_imus)
        print(f"Global steady state region identified: indices {start_idx} to {end_idx}")
    except ValueError as e:
        print(f"Failed to find global steady state: {e}")
        return {}
    
    for link, imus in link_mapping.items():
        print(f"Processing Link: {link} with IMUs {imus}")
        
        # Check if all IMUs for this link exist in the DataFrame
        missing_imus = [imu for imu in imus if f'{imu}_AX' not in df.columns]
        if missing_imus:
            print(f"  -> Skipping {link} because columns for {missing_imus} were not found.")
            continue
            
        N = len(imus)
        
        # 1. Alignment to Global Reference
        R_offsets_dict, R_world_to_master = compute_apose_alignments(global_steady_means, imus)
        print(f"  -> Rotation offsets:")
        for imu_name, r_off in R_offsets_dict.items():
            print(f"       {imu_name}:\n{r_off}")
            
        R_offsets = [R_offsets_dict[imu] for imu in imus]
        
        # 2. Initialize EKF State (R_k tracks Master IMU to World)
        R_k = R_world_to_master.T.copy() 
        b_k_list = [global_steady_means[imu]['gyr_mean'] for imu in imus] # Initialize bias from steady state
        
        state_dim = 3 + 3*N
        P_k = np.eye(state_dim) * 0.1
        
        link_rotations = []
        link_euler = []
        link_variance = []
        
        # 3. Time Loop
        for index, row in df.iterrows():
            acc_meas_list = []
            gyr_meas_list = []
            
            for imu in imus:
                acc_cols = [f'{imu}_AX', f'{imu}_AY', f'{imu}_AZ']
                gyr_cols = [f'{imu}_GX', f'{imu}_GY', f'{imu}_GZ']
                
                acc_meas_list.append(row[acc_cols].values.astype(float))
                # Convert Gyro from degrees/sec to radians/sec
                gyr_meas_list.append(row[gyr_cols].values.astype(float) * (np.pi / 180.0))
                
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
            link_euler.append(r.as_euler('xyz', degrees=True))
            link_variance.append(np.trace(P_k[:3, :3]))
            
        link_orientations[link] = {
            'quaternions': link_rotations,
            'euler_angles': link_euler,
            'orientation_variance': link_variance,
            'R_world_to_master': R_world_to_master
        }
        print(f"  -> Successfully processed {len(link_rotations)} frames for {link}.")
        
    return link_orientations


# =====================================================================
# 5. 3D Kinematics Animation
# =====================================================================

def animate_arm_3d(link_orientations, step=10, dt=0.01):
    """
    Creates a 3D Plotly animation of the arm (Upper Arm, Lower Arm, Palm)
    connected as a kinematic chain. Plays back in real-time speed.
    """
    if not all(link in link_orientations for link in ['Upper_Arm_Right', 'Lower_Arm_Right', 'Palm_Right']):
        print("Missing links for full arm animation.")
        return
        
    print("Generating 3D animation (this might take a moment)...")
    
    # Bone lengths (meters)
    L_upper = 0.30
    L_lower = 0.25
    L_palm  = 0.10
    
    # Extract quaternions [w, x, y, z] -> [x, y, z, w] for SciPy
    def convert_quat(q_list):
        return np.array([[q[1], q[2], q[3], q[0]] for q in q_list])
        
    q_upper = convert_quat(link_orientations['Upper_Arm_Right']['quaternions'])
    q_lower = convert_quat(link_orientations['Lower_Arm_Right']['quaternions'])
    q_palm  = convert_quat(link_orientations['Palm_Right']['quaternions'])
    
    min_len = min(len(q_upper), len(q_lower), len(q_palm))
    indices = np.arange(0, min_len, step)
    
    # Assuming Z is against gravity, bone points along -Z in the global A-pose frame
    v_upper_world = np.array([0, 0, -L_upper])
    v_lower_world = np.array([0, 0, -L_lower])
    v_palm_world  = np.array([0, 0, -L_palm])
    
    # Transform bone vectors to the local IMU frame using their A-pose orientations
    v_upper = link_orientations['Upper_Arm_Right']['R_world_to_master'] @ v_upper_world
    v_lower = link_orientations['Lower_Arm_Right']['R_world_to_master'] @ v_lower_world
    v_palm  = link_orientations['Palm_Right']['R_world_to_master'] @ v_palm_world
    
    p0 = np.zeros((len(indices), 3)) # Shoulder
    p1 = np.zeros((len(indices), 3)) # Elbow
    p2 = np.zeros((len(indices), 3)) # Wrist
    p3 = np.zeros((len(indices), 3)) # Fingertips
    
    for i, idx in enumerate(indices):
        R_up = R.from_quat(q_upper[idx]).as_matrix()
        R_low = R.from_quat(q_lower[idx]).as_matrix()
        R_pa = R.from_quat(q_palm[idx]).as_matrix()
        
        p1[i] = p0[i] + R_up @ v_upper
        p2[i] = p1[i] + R_low @ v_lower
        p3[i] = p2[i] + R_pa @ v_palm
        
    # Static Body reference (Right Shoulder -> Neck -> Head, Neck -> Pelvis, Neck -> Left Shoulder)
    torso_x = [0, 0, 0, 0, 0, 0, 0]
    torso_y = [0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2]
    torso_z = [0, 0, 0.2, 0, -0.6, 0, 0]
    
    torso_trace = go.Scatter3d(
        x=torso_x, y=torso_y, z=torso_z,
        mode='lines+markers',
        marker=dict(size=4, color='gray'),
        line=dict(color='gray', width=8),
        name='Static Torso'
    )
    
    # Static Floor
    floor_trace = go.Surface(
        x=[[-1, 1], [-1, 1]],
        y=[[-1, -1], [1, 1]],
        z=[[-0.8, -0.8], [-0.8, -0.8]],
        colorscale='Greys',
        showscale=False,
        opacity=0.3,
        name='Floor'
    )
    
    arm_trace = go.Scatter3d(
        x=[p0[0,0], p1[0,0], p2[0,0], p3[0,0]],
        y=[p0[0,1], p1[0,1], p2[0,1], p3[0,1]],
        z=[p0[0,2], p1[0,2], p2[0,2], p3[0,2]],
        mode='lines+markers',
        marker=dict(size=6, color=['black', 'red', 'green', 'blue']),
        line=dict(color='blue', width=6),
        name='Right Arm'
    )

    # Create the Plotly animation
    fig = go.Figure(
        data=[arm_trace, torso_trace, floor_trace],
        layout=go.Layout(
            title="3D Arm Kinematics Animation",
            scene=dict(
                xaxis=dict(range=[-0.5, 0.5], title='X (m) (Anterior/Posterior)'),
                yaxis=dict(range=[-0.5, 0.5], title='Y (m) (Lateral/Medial)'),
                zaxis=dict(range=[-0.8, 0.5], title='Z (m) (Vertical)'),
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=int(step*dt*1000), redraw=True), fromcurrent=True)])]
            )]
        ),
        frames=[
            go.Frame(
                data=[go.Scatter3d(
                    x=[p0[i,0], p1[i,0], p2[i,0], p3[i,0]],
                    y=[p0[i,1], p1[i,1], p2[i,1], p3[i,1]],
                    z=[p0[i,2], p1[i,2], p2[i,2], p3[i,2]]
                )],
                name=str(i),
                traces=[0] # Only update the arm trace (index 0)
            ) for i in range(len(indices))
        ]
    )
    fig.show()

# =====================================================================
# Execution & Example Plotting (from user request)
# =====================================================================
if __name__ == "__main__":
    # Load the specific dataset
    df_path = "/home/jazz/Projects/Wearable-Biomechanics/experiments_0001_sub_0001_1778229520.csv"
    try:
        df = pd.read_csv(df_path)
        print(f"Loaded dataset with shape: {df.shape}")

        link_orientations = process_dataset(df, dt=0.01) # Set your actual dt here
        
        # Plot EKF Estimation and Variance using Plotly
        for link, data in link_orientations.items():
            if len(data['quaternions']) > 0:
                # Plot Quaternions
                quat_df = pd.DataFrame(data['quaternions'], columns=['W', 'X', 'Y', 'Z'])
                if 'Timestamp' in df.columns:
                    quat_df['Timestamp'] = df['Timestamp'].values
                else:
                    quat_df['Timestamp'] = np.arange(len(quat_df)) * 0.01
                    
                fig_quat = px.line(
                    quat_df, 
                    x='Timestamp', 
                    y=['W', 'X', 'Y', 'Z'], 
                    title=f'{link} - EKF Estimated Quaternions Over Time (Gimbal Lock Free)'
                )
                fig_quat.show()
                
                # Plot Variance
                var_df = pd.DataFrame({
                    'Orientation Variance (Trace)': data['orientation_variance'], 
                    'Timestamp': quat_df['Timestamp']
                })
                fig_var = px.line(
                    var_df,
                    x='Timestamp',
                    y='Orientation Variance (Trace)',
                    title=f'{link} - EKF Orientation Variance Over Time'
                )
                fig_var.show()
                
                # Break after plotting the first link to avoid opening too many browser tabs
                break
                
        # Generate 3D Animation
        animate_arm_3d(link_orientations, step=15) # Downsample to make it faster
        
    except FileNotFoundError:
        print(f"Could not find dataset at: {df_path}")
