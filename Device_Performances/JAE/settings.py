"""
settings.py

Initiated: 20/04/2024
Updated:   22/07/2025

Defines global settings (e.g. constants, etc.) for the analysis.
This is a top-level code: this will not use any lower level code.
"""

import numpy as np

#region Absolute constants ---------------------------------------------------------------------------------------

N_3D_COMPONENTS   = 3
N_QUAT_COMPONENTS = 4                       # Number of components in a quaternion

g                 = 9.80665                 # m/s^2  # Standard gravitational acceleration
pi                = np.pi # Constant Pi
g_earth_vector    = [0, 0, -g]              # The standard gravity vector in the earth frame
xyz_axis_strings  = ('x', 'y', 'z')         # Axis strings for x, y, z axes
deg_to_rad        = np.pi/180.0             # Multiplier to convert degrees to radians
rad_to_deg        = 180.0/np.pi             # Multiplier to convert radians to degrees

#endregion -------------------------------------------------------------------------------------------------------

#region Kalman ---------------------------------------------------------------------------------------------------

# TODO: find these quantities on the run (during the dataset noise determination)
# Standard deviation of acceleration measurement noise (for continuous time)
# 180 micro-g/sqrt(Hz) converted to m/s^2 which is 180 * 9.81 * 1e-6
sigma_accel_m_c   = 180 * 9.81 * 1e-6

# Standard deviation of angular velocity rate noise (for continuous time)
# 0.007 deg/s/sqrt(Hz) converted to rad/s which is np.radians(0.007)
sigma_omega_r_c   = np.radians(0.007)

# Standard deviation of angular velocity bias noise (for continuous time)
# 10 deg/hr converted to rad/s which is np.radians(10) / 3600
sigma_omega_w_c   = np.radians(10) / 3600

#endregion -------------------------------------------------------------------------------------------------------

#region Paths ----------------------------------------------------------------------------------------------------

FFMPEGPath            = r'C:/Program Files/ffmpeg/bin/ffmpeg.exe' # Absolute path to FFMPEG executable
DatasetGenerationPath = r'dataset_gen/' # Relative path to where the generated datasets are to be written
ResultsOutputPath     = r'results_out/' # Relative path to where the generated results are to be written
YOLOModelPath         = r'yolo_models/v11/yolo11x-pose.pt' # Relative path to the YOLOvx model (.pt file)

#endregion -------------------------------------------------------------------------------------------------------