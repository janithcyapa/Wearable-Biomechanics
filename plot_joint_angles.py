import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from jae import estimate_joint_angles, flexion_angle

def get_sensor_data(df, node, imu):
    """Helper to extract and convert sensor data for a specific node/imu."""
    acc_cols_orbix = [f'OrbiX-Node-{node}_IMU{imu}_AX', f'OrbiX-Node-{node}_IMU{imu}_AY', f'OrbiX-Node-{node}_IMU{imu}_AZ']
    gyr_cols_orbix = [f'OrbiX-Node-{node}_IMU{imu}_GX', f'OrbiX-Node-{node}_IMU{imu}_GY', f'OrbiX-Node-{node}_IMU{imu}_GZ']
    
    acc_cols_dev = [f'device{node}_IMU{imu}_AX', f'device{node}_IMU{imu}_AY', f'device{node}_IMU{imu}_AZ']
    gyr_cols_dev = [f'device{node}_IMU{imu}_GX', f'device{node}_IMU{imu}_GY', f'device{node}_IMU{imu}_GZ']
    
    # jae.py expects units in g, assuming the data is in m/s^2, we divide by 9.81
    if all(col in df.columns for col in acc_cols_orbix + gyr_cols_orbix):
        acc = df[acc_cols_orbix].values / 9.81
        gyr = df[gyr_cols_orbix].values
        return acc, gyr
    elif all(col in df.columns for col in acc_cols_dev + gyr_cols_dev):
        acc = df[acc_cols_dev].values / 9.81
        gyr = df[gyr_cols_dev].values
        return acc, gyr
    else:
        return None, None

def process_joint(df, node_prox, imu_prox, node_dist, imu_dist, dt=0.01):
    """Extracts data and computes Euler angles for a specific joint."""
    acc_prox, gyr_prox = get_sensor_data(df, node_prox, imu_prox)
    acc_dist, gyr_dist = get_sensor_data(df, node_dist, imu_dist)
    
    if acc_prox is None or acc_dist is None:
        return None
    
    # Convert Gyroscope to rad/s if it appears to be in degrees/s (values > 50 are a good hint)
    if np.max(np.abs(gyr_prox)) > 50 or np.max(np.abs(gyr_dist)) > 50:
        gyr_prox = np.deg2rad(gyr_prox)
        gyr_dist = np.deg2rad(gyr_dist)

    result = estimate_joint_angles(
        acc_prox=acc_prox,
        gyr_prox=gyr_prox,
        acc_dist=acc_dist,
        gyr_dist=gyr_dist,
        dt=dt,
        n_cal_samples=100,      # 1 second of static posture at the beginning
        acc_gate_threshold=0.3,
        euler_seq="xyz"
    )
    return result.euler_swing

def analyze_and_plot(csv_path="test_dataset.csv", dt=0.01):
    """
    Reads the dataset, extracts IMU data for multiple joints, 
    estimates joint angles, and returns a Plotly Dash app.
    """
    df = pd.read_csv(csv_path)
    timestamps = df['Timestamp'].values if 'Timestamp' in df.columns else np.arange(len(df))
    
    # Define the joints and their corresponding Proximal/Distal sensors
    all_joints = {
        "Right Knee": {"prox": (1, 2), "dist": (10, 1)},
        "Left Knee":  {"prox": (3, 2), "dist": (8, 1)},
        "Right Elbow":{"prox": (2, 2), "dist": (6, 1)},
        "Left Elbow": {"prox": (4, 2), "dist": (7, 1)},
    }
    
    # Filter joints that have available data
    joints = {}
    for name, nodes in all_joints.items():
        # Check if data exists for this joint
        acc_prox, _ = get_sensor_data(df, nodes["prox"][0], nodes["prox"][1])
        acc_dist, _ = get_sensor_data(df, nodes["dist"][0], nodes["dist"][1])
        if acc_prox is not None and acc_dist is not None:
            joints[name] = nodes

    if not joints:
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Wearable Biomechanics - No Data Available", style={'textAlign': 'center', 'fontFamily': 'sans-serif'})
        ])
        return app
    
    # Create subplots based on available joints
    n_joints = len(joints)
    fig = make_subplots(
        rows=n_joints, cols=1, 
        subplot_titles=list(joints.keys()),
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    for i, (name, nodes) in enumerate(joints.items()):
        row = i + 1
        col = 1
        
        euler_swing = process_joint(df, nodes["prox"][0], nodes["prox"][1], nodes["dist"][0], nodes["dist"][1], dt)
        
        # Only show legend for the first subplot to prevent duplication
        show_leg = (row == 1)
        
        fig.add_trace(go.Scatter(x=timestamps, y=euler_swing[:, 0], mode='lines', name='Roll (x)', legendgroup='roll', line=dict(color='#EF553B'), showlegend=show_leg), row=row, col=col)
        fig.add_trace(go.Scatter(x=timestamps, y=euler_swing[:, 1], mode='lines', name='Pitch (y) - Flexion', legendgroup='pitch', line=dict(color='#00CC96'), showlegend=show_leg), row=row, col=col)
        fig.add_trace(go.Scatter(x=timestamps, y=euler_swing[:, 2], mode='lines', name='Yaw (z)', legendgroup='yaw', line=dict(color='#AB63FA'), showlegend=show_leg), row=row, col=col)
        
        # Y-axis titles for all rows (since it's only 1 column)
        fig.update_yaxes(title_text="Angle (°)", row=row, col=col)
        
        # Force tick labels on all x-axes
        fig.update_xaxes(showticklabels=True, row=row, col=col)
        
        # X-axis title for bottom row
        if row == n_joints:
            fig.update_xaxes(title_text="Timestamp / Sample", row=row, col=col)
            
    fig.update_layout(
        title="Joint Angle Estimation - Multiple Joints",
        template="plotly_dark",
        hovermode="x unified",
        height=850
    )
    
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Wearable Biomechanics - Multi-Joint Angles", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),
        dcc.Graph(figure=fig, style={'height': '90vh'})
    ])
    
    return app

if __name__ == "__main__":
    # app = analyze_and_plot("test_dataset.csv", dt=0.01)
    app = analyze_and_plot("./Device_Performances/JAE/entry_0001_none_1781856275.csv", dt=0.01)

    # Run the web server
    app.run(debug=True, port=8050)
