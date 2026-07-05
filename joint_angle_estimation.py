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
    'Pelvis': ['device5_IMU1','device5_IMU2','device5_IMU3'],
    'Torso': ['device9_IMU1'],
    'Shoulder_Right': ['device9_IMU2'],
    'Shoulder_Left': ['device9_IMU3'],
    'Upper_Arm_Right': ['device2_IMU1', 'device2_IMU2'],
    'Lower_Arm_Right': ['device6_IMU1', 'device6_IMU2'],
    'Palm_Right': ['device6_IMU3'],
    'Upper_Arm_Left': ['device4_IMU1','device4_IMU2'],
    'Lower_Arm_Left': ['device7_IMU1','device7_IMU2'],
    'Palm_Left': ['device7_IMU3'],
    'Upper_Leg_Right': ['device1_IMU1','device1_IMU2'],
    'Lower_Leg_Right': ['device10_IMU1','device10_IMU2'],
    'Foot_Right': ['device10_IMU3'],
    'Upper_Leg_Left': ['device3_IMU1','device3_IMU2'],
    'Lower_Leg_Left': ['device8_IMU1','device8_IMU2'],
    'Foot_Left': ['device8_IMU3'],
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

def compute_apose_alignments(steady_means, imu_cols, link_name):
    """
    Computes R_offset for all IMUs relative to the first IMU (Master) 
    based on the A-Pose geometry, accounting for limb-specific orientations.
    """
    R_initials = {}
    
    for imu in imu_cols:
        acc_mean = steady_means[imu]['acc_mean']
        
        # Z-axis (Against gravity) is Up
        Z = acc_mean / np.linalg.norm(acc_mean)
        
        # Segment-specific Tie-Breaker for Yaw
        # We adjust our 'assumed forward' or 'assumed lateral' based on the limb
        if 'Leg' in link_name:
            # For legs, assuming the sensor X-axis generally points DOWN the leg
            # and we want the person facing Global X.
            ref_vector = np.array([1.0, 0, 0]) # Force X to align with Global Forward
            Y = np.cross(Z, ref_vector)
            
            if np.linalg.norm(Y) < 1e-5:
                ref_vector = np.array([0, -1.0, 0])
                Y = np.cross(Z, ref_vector)
                
            Y = Y / np.linalg.norm(Y)
            X = np.cross(Y, Z)
            
        else:
            # Original logic for Arms/Torso
            ref_Y = np.array([0, 1.0, 0]) 
            X = np.cross(ref_Y, Z)
            
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
        R_offsets_dict, R_world_to_master = compute_apose_alignments(global_steady_means, imus, link)
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
# 5. Full-Body 3D Kinematics Animation
# =====================================================================

# Skeleton definition: bone lengths (meters) and kinematic chain
BONE_LENGTHS = {
    'Pelvis':           0.15,  # Pelvis height (root)
    'Torso':            0.40,  # Pelvis to neck
    'Shoulder_Right':   0.18,  # Neck to right shoulder joint
    'Shoulder_Left':    0.18,  # Neck to left shoulder joint
    'Upper_Arm_Right':  0.28,  # Shoulder to elbow
    'Lower_Arm_Right':  0.25,  # Elbow to wrist
    'Palm_Right':       0.10,  # Wrist to fingertips
    'Upper_Arm_Left':   0.28,
    'Lower_Arm_Left':   0.25,
    'Palm_Left':        0.10,
    'Upper_Leg_Right':  0.42,  # Hip to knee
    'Lower_Leg_Right':  0.40,  # Knee to ankle
    'Foot_Right':       0.12,  # Ankle to toe
    'Upper_Leg_Left':   0.42,
    'Lower_Leg_Left':   0.40,
    'Foot_Left':        0.12,
}

# A-pose bone directions in world frame (Z up, X forward, Y left)
# Arms hang slightly to the sides, legs point down, torso points up
BONE_DIRECTIONS = {
    'Pelvis':           np.array([0, 0, 0]),         # Root origin (no bone vector)
    'Torso':            np.array([0, 0, 1]),          # Up
    'Shoulder_Right':   np.array([0, -1, 0]),         # Right
    'Shoulder_Left':    np.array([0, 1, 0]),          # Left
    'Upper_Arm_Right':  np.array([0, 0, -1]),         # Down (arms at sides in A-pose)
    'Lower_Arm_Right':  np.array([0, 0, -1]),         # Down
    'Palm_Right':       np.array([0, 0, -1]),         # Down
    'Upper_Arm_Left':   np.array([0, 0, -1]),         # Down
    'Lower_Arm_Left':   np.array([0, 0, -1]),         # Down
    'Palm_Left':        np.array([0, 0, -1]),         # Down
    'Upper_Leg_Right':  np.array([0, 0, -1]),         # Down
    'Lower_Leg_Right':  np.array([0, 0, -1]),         # Down
    'Foot_Right':       np.array([1, 0, 0]),          # Forward
    'Upper_Leg_Left':   np.array([0, 0, -1]),         # Down
    'Lower_Leg_Left':   np.array([0, 0, -1]),         # Down
    'Foot_Left':        np.array([1, 0, 0]),          # Forward
}

# Kinematic chain: child -> parent
# Each child's base position = parent's end position
KINEMATIC_CHAIN = {
    'Pelvis':           None,           # Root
    'Torso':            'Pelvis',
    'Shoulder_Right':   'Torso',
    'Shoulder_Left':    'Torso',
    'Upper_Arm_Right':  'Shoulder_Right',
    'Lower_Arm_Right':  'Upper_Arm_Right',
    'Palm_Right':       'Lower_Arm_Right',
    'Upper_Arm_Left':   'Shoulder_Left',
    'Lower_Arm_Left':   'Upper_Arm_Left',
    'Palm_Left':        'Lower_Arm_Left',
    'Upper_Leg_Right':  'Pelvis',
    'Lower_Leg_Right':  'Upper_Leg_Right',
    'Foot_Right':       'Lower_Leg_Right',
    'Upper_Leg_Left':   'Pelvis',
    'Lower_Leg_Left':   'Upper_Leg_Left',
    'Foot_Left':        'Lower_Leg_Left',
}

# Topological order for forward kinematics traversal
CHAIN_ORDER = [
    'Pelvis', 'Torso',
    'Shoulder_Right', 'Upper_Arm_Right', 'Lower_Arm_Right', 'Palm_Right',
    'Shoulder_Left', 'Upper_Arm_Left', 'Lower_Arm_Left', 'Palm_Left',
    'Upper_Leg_Right', 'Lower_Leg_Right', 'Foot_Right',
    'Upper_Leg_Left', 'Lower_Leg_Left', 'Foot_Left',
]

# Skeleton line segments for drawing (pairs of link names to connect)
SKELETON_SEGMENTS = [
    # Spine
    ('Pelvis', 'Torso'),
    # Right arm
    ('Torso', 'Shoulder_Right'),
    ('Shoulder_Right', 'Upper_Arm_Right'),
    ('Upper_Arm_Right', 'Lower_Arm_Right'),
    ('Lower_Arm_Right', 'Palm_Right'),
    # Left arm
    ('Torso', 'Shoulder_Left'),
    ('Shoulder_Left', 'Upper_Arm_Left'),
    ('Upper_Arm_Left', 'Lower_Arm_Left'),
    ('Lower_Arm_Left', 'Palm_Left'),
    # Right leg
    ('Pelvis', 'Upper_Leg_Right'),
    ('Upper_Leg_Right', 'Lower_Leg_Right'),
    ('Lower_Leg_Right', 'Foot_Right'),
    # Left leg
    ('Pelvis', 'Upper_Leg_Left'),
    ('Upper_Leg_Left', 'Lower_Leg_Left'),
    ('Lower_Leg_Left', 'Foot_Left'),
]

# Colors per body region
SEGMENT_COLORS = {
    ('Pelvis', 'Torso'):                  '#FFD700',  # Gold - spine
    ('Torso', 'Shoulder_Right'):          '#FF6B6B',  # Red - right
    ('Shoulder_Right', 'Upper_Arm_Right'):'#FF6B6B',
    ('Upper_Arm_Right', 'Lower_Arm_Right'):'#FF4444',
    ('Lower_Arm_Right', 'Palm_Right'):    '#CC0000',
    ('Torso', 'Shoulder_Left'):           '#4ECDC4',  # Teal - left
    ('Shoulder_Left', 'Upper_Arm_Left'):  '#4ECDC4',
    ('Upper_Arm_Left', 'Lower_Arm_Left'): '#2EAD9E',
    ('Lower_Arm_Left', 'Palm_Left'):      '#1A7A6F',
    ('Pelvis', 'Upper_Leg_Right'):        '#FF8C42',  # Orange - right leg
    ('Upper_Leg_Right', 'Lower_Leg_Right'):'#E07020',
    ('Lower_Leg_Right', 'Foot_Right'):    '#C05000',
    ('Pelvis', 'Upper_Leg_Left'):         '#6C5CE7',  # Purple - left leg
    ('Upper_Leg_Left', 'Lower_Leg_Left'): '#5A42D0',
    ('Lower_Leg_Left', 'Foot_Left'):      '#4830B0',
}


def animate_body_3d(link_orientations, step=10, dt=0.01):
    """
    Creates a 3D Plotly animation of the full body skeleton.
    Uses forward kinematics from the Pelvis root through all kinematic chains.
    """
    available_links = [link for link in CHAIN_ORDER if link in link_orientations 
                       and len(link_orientations[link]['quaternions']) > 0]
    
    if len(available_links) == 0:
        print("No links available for animation.")
        return
    
    print(f"Generating full-body 3D animation with {len(available_links)} links...")
    print(f"  Available: {available_links}")
    
    # Convert quaternions [w,x,y,z] -> [x,y,z,w] for SciPy
    def convert_quat(q_list):
        return np.array([[q[1], q[2], q[3], q[0]] for q in q_list])
    
    # Prepare quaternion arrays and bone vectors for each available link
    quat_arrays = {}
    bone_vectors_local = {}  # bone vector in the IMU's local frame
    
    for link in available_links:
        quat_arrays[link] = convert_quat(link_orientations[link]['quaternions'])
        
        # Bone vector in world frame, transformed to local IMU frame
        direction = BONE_DIRECTIONS[link]
        length = BONE_LENGTHS[link]
        v_world = direction * length
        
        R_w2m = link_orientations[link]['R_world_to_master']
        bone_vectors_local[link] = R_w2m @ v_world
    
    # Determine the minimum frame count across all available links
    min_len = min(len(quat_arrays[link]) for link in available_links)
    indices = np.arange(0, min_len, step)
    n_frames = len(indices)
    
    # Pelvis origin height: place pelvis so feet are near ground in A-pose
    # Approximate: pelvis_z = upper_leg + lower_leg + foot_height
    pelvis_z = BONE_LENGTHS['Upper_Leg_Right'] + BONE_LENGTHS['Lower_Leg_Right'] + BONE_LENGTHS['Foot_Right'] * 0.3
    pelvis_origin = np.array([0, 0, pelvis_z])
    
    # Compute joint positions via forward kinematics for each frame
    # Each link has a 'start' (base) and 'end' (tip) position
    joint_starts = {link: np.zeros((n_frames, 3)) for link in available_links}
    joint_ends = {link: np.zeros((n_frames, 3)) for link in available_links}
    
    for fi, idx in enumerate(indices):
        # Forward kinematics pass
        for link in CHAIN_ORDER:
            if link not in available_links:
                continue
            
            parent = KINEMATIC_CHAIN[link]
            
            if parent is None:
                # Root (Pelvis): starts at the pelvis origin
                joint_starts[link][fi] = pelvis_origin
            elif parent in available_links:
                # Start where the parent ends
                joint_starts[link][fi] = joint_ends[parent][fi]
            else:
                # Parent not available; fall back to pelvis
                joint_starts[link][fi] = pelvis_origin
            
            # Apply orientation to bone vector
            R_mat = R.from_quat(quat_arrays[link][idx]).as_matrix()
            joint_ends[link][fi] = joint_starts[link][fi] + R_mat @ bone_vectors_local[link]
    
    # ------------------------------------------------------------------
    # Build Plotly figure
    # ------------------------------------------------------------------
    
    def build_skeleton_traces(fi):
        """Build line traces for one frame."""
        traces = []
        for seg_idx, (parent_link, child_link) in enumerate(SKELETON_SEGMENTS):
            if parent_link not in available_links or child_link not in available_links:
                continue
            
            # Connect parent's end to child's end via child's start
            p_end = joint_ends[parent_link][fi]
            c_start = joint_starts[child_link][fi]
            c_end = joint_ends[child_link][fi]
            
            color = SEGMENT_COLORS.get((parent_link, child_link), '#888888')
            
            traces.append(go.Scatter3d(
                x=[p_end[0], c_start[0], c_end[0]],
                y=[p_end[1], c_start[1], c_end[1]],
                z=[p_end[2], c_start[2], c_end[2]],
                mode='lines+markers',
                marker=dict(size=4, color=color),
                line=dict(color=color, width=6),
                name=f'{parent_link}→{child_link}',
                showlegend=(fi == 0),
            ))
        return traces
    
    # Joint markers for all links (large dots at joint positions)
    def build_joint_markers(fi):
        """Build joint marker trace for one frame."""
        xs, ys, zs, labels = [], [], [], []
        for link in available_links:
            # Start joint
            xs.append(joint_starts[link][fi][0])
            ys.append(joint_starts[link][fi][1])
            zs.append(joint_starts[link][fi][2])
            labels.append(link + ' (start)')
            # End joint
            xs.append(joint_ends[link][fi][0])
            ys.append(joint_ends[link][fi][1])
            zs.append(joint_ends[link][fi][2])
            labels.append(link + ' (end)')
        
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(size=3, color='white', line=dict(width=1, color='black')),
            text=labels,
            hoverinfo='text',
            name='Joints',
            showlegend=False,
        )
    
    # Floor grid
    floor_z = -0.05
    floor_trace = go.Surface(
        x=[[-1.5, 1.5], [-1.5, 1.5]],
        y=[[-1.5, -1.5], [1.5, 1.5]],
        z=[[floor_z, floor_z], [floor_z, floor_z]],
        colorscale=[[0, 'rgba(200,200,200,0.3)'], [1, 'rgba(200,200,200,0.3)']],
        showscale=False,
        opacity=0.2,
        name='Floor',
        hoverinfo='skip',
    )
    
    # Initial frame traces
    initial_traces = build_skeleton_traces(0) + [build_joint_markers(0), floor_trace]
    
    # Count how many traces are skeleton segments (to know which to update per frame)
    n_segment_traces = len(build_skeleton_traces(0))
    # Total animated traces = n_segment_traces + 1 (joint markers)
    animated_trace_indices = list(range(n_segment_traces + 1))
    
    # Build animation frames
    frames = []
    for fi in range(n_frames):
        frame_traces = build_skeleton_traces(fi) + [build_joint_markers(fi)]
        time_sec = indices[fi] * dt
        frames.append(go.Frame(
            data=frame_traces,
            name=f't={time_sec:.2f}s',
            traces=animated_trace_indices,
        ))
    
    # Axis range
    ax_range = 1.2
    
    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title="Full-Body 3D Kinematics Animation",
            scene=dict(
                xaxis=dict(range=[-ax_range, ax_range], title='X (m) (Forward)'),
                yaxis=dict(range=[-ax_range, ax_range], title='Y (m) (Lateral)'),
                zaxis=dict(range=[-0.2, 2.0], title='Z (m) (Vertical)'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.05,
                x=0.5,
                xanchor='center',
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=int(step * dt * 1000), redraw=True),
                                          fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate', transition=dict(duration=0))]),
                ],
            )],
            sliders=[dict(
                active=0,
                yanchor='top', xanchor='left',
                currentvalue=dict(prefix='Time: ', visible=True),
                pad=dict(b=10, t=50),
                len=0.9, x=0.05,
                steps=[
                    dict(args=[[f't={indices[fi]*dt:.2f}s'],
                               dict(frame=dict(duration=0, redraw=True),
                                    mode='immediate', transition=dict(duration=0))],
                         method='animate',
                         label=f'{indices[fi]*dt:.1f}s')
                    for fi in range(n_frames)
                ],
            )],
        ),
        frames=frames,
    )
    
    fig.show()


# =====================================================================
# Execution & Plotting
# =====================================================================
if __name__ == "__main__":
    # Load the specific dataset
    df_path = "/home/jazz/Projects/Wearable-Biomechanics/experiments_0001_sub_0001_1778229520.csv"
    try:
        df = pd.read_csv(df_path)
        print(f"Loaded dataset with shape: {df.shape}")

        link_orientations = process_dataset(df, dt=0.01)
        
        # Plot EKF Quaternion and Variance for each successfully processed link
        for link, data in link_orientations.items():
            if not data or len(data['quaternions']) == 0:
                continue
                
            # Build timestamp column
            if 'Timestamp' in df.columns:
                timestamps = df['Timestamp'].values[:len(data['quaternions'])]
            else:
                timestamps = np.arange(len(data['quaternions'])) * 0.01
            
            # Plot Quaternions
            quat_df = pd.DataFrame(data['quaternions'], columns=['W', 'X', 'Y', 'Z'])
            quat_df['Timestamp'] = timestamps
            fig_quat = px.line(
                quat_df, x='Timestamp', y=['W', 'X', 'Y', 'Z'],
                title=f'{link} - EKF Estimated Quaternions Over Time'
            )
            fig_quat.show()
            
            # Plot Variance
            var_df = pd.DataFrame({
                'Orientation Variance (Trace)': data['orientation_variance'],
                'Timestamp': timestamps
            })
            fig_var = px.line(
                var_df, x='Timestamp', y='Orientation Variance (Trace)',
                title=f'{link} - EKF Orientation Variance Over Time'
            )
            fig_var.show()
        
        # Generate Full-Body 3D Animation
        animate_body_3d(link_orientations, step=15, dt=0.01)
        
    except FileNotFoundError:
        print(f"Could not find dataset at: {df_path}")
