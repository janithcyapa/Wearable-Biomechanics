"""
kalman.py

Initiated: 20/04/2024
Updated:   08/10/2025

Defines a single step of each orientation estimation/denoising algorithm (e.g. DSS, IKF, UKF, EKF, etc.).
"""

import numpy as np
from collections import namedtuple
from numpy.linalg import norm

import settings as sts
import geometry as geo

# Direct state space integrator (gyro only)
def dss_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k, init_gravity_vector):
    return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1 K_kp1")
    return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
                                   P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None, K_kp1 = None)

    b_hat_kp1_k     = b_hat_k_k # Propagate the bias from previous step
    omega_hat_kp1_k = omega_m_kp1 - b_hat_kp1_k

    # Propagate quaternion using the quaternion from previous step q_hat_k_k
    OMEGA = geo.cross_prod_mat_4

    # Zeroth order integrator
    # q_hat_kp1_k = (geo.mat_exp(0.5*OMEGA(omega_hat_k_k)*dt)) @ q_hat_k_k

    # First order integrator
    omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
    q_hat_kp1_k = geo.mat_exp(0.5*OMEGA(omega_bar)*dt) @ q_hat_k_k
    q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k

    # Convert the quaternion to a rotation matrix
    C_kp1 = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)

    # Unused variables
    P_kp1_k = np.zeros(shape=(6,6))
    K_kp1   = np.zeros(shape=(6,3))
    dq_plus = np.zeros(shape=(4,1))

    # Populate return list
    return_list = return_list_type(C_kp1, q_hat_kp1_k, 0*dq_plus, 0*P_kp1_k, b_hat_kp1_k, omega_hat_kp1_k, K_kp1)

    return return_list
#enddef

def ikf_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k, init_gravity_vector):
    """
    Applies a step of Indirect Kalman filtering

    Parameters
    ---------------
    P_k_k : 6x6 matrix
        State covariance matrix of (k-1)th (previous) step
    dt : float
        Sampling interval
    omega_m_kp1 : vector
        Angular velocity vector
    acc_r : vector
        Acceleration vector
    q_hat_k_k : quaternion
        Quaternion of rotation at (k-1)th (previous) step
    b_hat_k_k : vector
        Gyro noise bias at (k-1)th (previous) step
    sigma_r : float
        Standard deviation of angular velocity noise
    sigma_b : float
        Standard deviation of gyro noise bias
    omega_hat_k_k : vector
        Angular velocity vector of the previous step
    init_gravity_vector : vector
        Initial gravity vector, used to align the sensors with the direction of gravity
    
    Returns
    -------
    namedtuple
        A namedtuple object containing:
            C_kp1         : Rotation matrix at kth (current) sample
            q_kp1_kp1     : Quaternion of rotation at kth (current) sample
            dq_plus       : dq of quaternion of rotation at kth (current) sample
            P_kp1_kp1     : State covariance matrix at kth (current) sample
            b_hat_kp1_kp1 : Gyro noise bias at kth sample (current) sample
        A namedtuple is defined to distinguish easily between the components being returned.
    """

    return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1 K_kp1")
    return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
                                   P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None, K_kp1 = None)
    
    gravity_vector = init_gravity_vector # Gravity in the +x direction
                                         # as the sensors are oriented such that +x aligns with g

    # Assume that noise is equally distributed in the three spatial directions
    # Noise covariance matrix becomes and identity matrix
    Nr = (sts.sigma_omega_r_c**2)*geo.eye3 # Noise covariance for angular velocity noise
    Nw = (sts.sigma_omega_w_c**2)*geo.eye3 # Noise covariance matrix for angular velocity bias rate noise
    R  = (sts.sigma_accel_m_c**2)*geo.eye3 # Noise covariance matrix for acceleration measurement
    Qc = np.block([[Nr,geo.z3],[geo.z3,Nw]])

    b_hat_kp1_k     = b_hat_k_k # Propagate the bias from previous step
    omega_hat_kp1_k = omega_m_kp1 - b_hat_kp1_k

    # Propagate quaternion using the quaternion from previous step q_hat_k_k
    OMEGA = geo.cross_prod_mat_4

    # Zeroth order integrator
    q_hat_kp1_k = (geo.mat_exp(0.5*OMEGA(omega_hat_k_k)*dt)) @ q_hat_k_k

    # First order integrator
    # omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
    # q_hat_kp1_k = geo.mat_exp(0.5*OMEGA(omega_bar)*dt) @ q_hat_k_k
    # q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k

    omega0 = omega_hat_kp1_k

    # System matrix for error state space (continuous-time) (constant)
    Fc = np.block([[-geo.cross_prod_mat_3(omega_hat_k_k), -geo.eye3], 
                   [ geo.z3,                               geo.z3  ]])
    
    Gc = np.block([[-geo.eye3, geo.z3  ],
                   [ geo.z3,   geo.eye3]])
    
    # Approximation for Qd_k
    # Q11 = (sigma_omega_r_c**2*dt*geo.eye3) + sigma_omega_w_c**2 * ((geo.eye3 * ((dt**3)/3)) + (((((norm(omega0) * dt)**3)/3) + 2*np.sin(norm(omega0)*dt) - (2*norm(omega0)*dt) )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**5)))
    # Q22 = (sigma_omega_w_c**2)*dt*geo.eye3
    # Q12 = -(sigma_omega_w_c**2) * ( (geo.eye3*((dt**2)/2)) - ((((norm(omega0)*dt) - np.sin(norm(omega0)*dt))/(norm(omega0))**3)*geo.cross_prod_mat_3(omega0)) + (((((norm(omega0) * dt)**2)/2) + np.cos(norm(omega0)*dt) - 1 )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**4)))
    # Q21 = Q12.transpose()
    # Qd_k_ = np.block([[Q11, Q12],
    #                   [Q21, Q22]])

    # State transition matrix for the discrete time error state space
    # Phi_kp1 = geo.mat_exp(Fc*dt)
    # Phi_kp1 = Fc*dt
    # Phi_kp1 = Fc*dt + 0.5*(Fc*dt)@(Fc*dt)
    Phi_kp1 = geo.mat_exp(Fc*dt) - geo.eye6 # TODO: Why eye6?

    # Compute discrete-time process noise covariance matrix
    Qd_k = Phi_kp1 @ Gc @ Qc @ Gc.transpose() @ Phi_kp1.transpose() * dt

    # Test if the approximation is correct
    # deltaQd = Qd_k - Qd_k_
    # deltaQd[np.abs(deltaQd) < 1e-8] = 0
    # assert(np.all(deltaQd == np.zeros(shape=(6,6))))

    # Predicted state covariance matrix
    P_kp1_k = Phi_kp1 @ P_k_k @ Phi_kp1.transpose() + Qd_k

    # Kalman Update --------------------------------------------------------------------------------
    # Given the propagated state estimates q_hat_kp1_k, b_hat_kp1_k, P_kp1_k, z_kp1 (current measurement), H (meas matrix)

    # Method to accompany acceleration: failed
    # R_q_hat_k_k = geo.quaternion_to_rotation_matrix(q_hat_k_k)
    # R_q_hat_kp1_k = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
    # a_hat_world = R_q_hat_k_k @ acc_r + gravity_vector
    # a_hat_m     = R_q_hat_kp1_k.T @ (a_hat_world - gravity_vector)
    # z_kp1 = acc_r - a_hat_m
    # H_kp1 = np.block([-R_q_hat_kp1_k.T @ geo.cross_prod_mat_3(a_hat_world - gravity_vector), geo.z3])
    # r_kp1 = z_kp1
    
    # Measurement matrix
    C = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
    H_kp1 = np.block([geo.cross_prod_mat_3(C.T @ gravity_vector), np.zeros(shape=(3,3))])
    # H_kp1 = np.block([geo.cross_prod_mat_3(C.T @ gravity_vector), geo.z3]) # TEST

    z_hat_kp1 = C.T @ gravity_vector

    # Compute residual
    z_kp1 = acc_r
    r_kp1 = z_kp1 - z_hat_kp1

    # Covariance of residual
    S_kp1 = H_kp1 @ P_kp1_k @ H_kp1.transpose() + R

    # Kalman gain
    K_kp1 = P_kp1_k @ H_kp1.transpose() @ np.linalg.inv(S_kp1)

    # Correction
    delta_theta_hat_plus = K_kp1 @ r_kp1

    delta_q_hat_plus = 0.5*delta_theta_hat_plus[0:3] # Error angle vector
    delta_b_hat_plus = delta_theta_hat_plus[3:6] # Gyro noise bias delta

    delta_q_plus_dot_prod = delta_q_hat_plus.transpose() @ delta_q_hat_plus

    if delta_q_plus_dot_prod <= 1:
        delta_quat_hat = np.block([[delta_q_hat_plus],[np.sqrt(1 - delta_q_plus_dot_prod)]])
    else:
        delta_quat_hat = (1/np.sqrt(1 + delta_q_plus_dot_prod))*np.block([[delta_q_hat_plus],[1]])
    #endif

    dq_plus = delta_quat_hat # Quaternion delta (correction to be applied)
    # dq_plus = np.block([[0.5*delta_x_hat_plus[0:3]],[np.array([1])]]) # Quaternion delta (correction to be applied)

    # Apply correction
    q_kp1_kp1 = geo.quaternion_multiply((dq_plus)/(norm(dq_plus)), q_hat_kp1_k) # Quaternion update
    b_hat_kp1_kp1 = b_hat_kp1_k + delta_b_hat_plus # Gyro noise bias update

    # Calculate rotation matrix corresponding to q_kp1_kp1
    C_kp1 = geo.quaternion_to_rotation_matrix(q_kp1_kp1)

    # Compute updated state covariance matrix
    P_kp1_kp1 = (geo.eye6-K_kp1@H_kp1) @ P_kp1_k @ (geo.eye6-K_kp1@H_kp1).transpose() + K_kp1 @ R @ K_kp1.transpose()

    # Update the estimated angular velocity using new estimation for bias
    omega_hat_kp1_kp1 = omega_m_kp1 - b_hat_kp1_kp1

    # Populate return list
    return_list = return_list_type(C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1, K_kp1)

    return return_list
#enddef

# def ikf_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k, init_gravity_vector):
#     """
#     Applies a step of Indirect Kalman filtering

#     Parameters
#     ---------------
#     P_k_k : 6x6 matrix
#         State covariance matrix of (k-1)th (previous) step
#     dt : float
#         Sampling interval
#     omega_m_kp1 : vector
#         Angular velocity vector
#     acc_r : vector
#         Acceleration vector
#     q_hat_k_k : quaternion
#         Quaternion of rotation at (k-1)th (previous) step
#     b_hat_k_k : vector
#         Gyro noise bias at (k-1)th (previous) step
#     sigma_r : float
#         Standard deviation of angular velocity noise
#     sigma_b : float
#         Standard deviation of gyro noise bias
#     omega_hat_k_k : vector
#         Angular velocity vector of the previous step
#     init_gravity_vector : vector
#         Initial gravity vector, used to align the sensors with the direction of gravity
    
#     Returns
#     -------
#     namedtuple
#         A namedtuple object containing:
#             C_kp1         : Rotation matrix at kth (current) sample
#             q_kp1_kp1     : Quaternion of rotation at kth (current) sample
#             dq_plus       : dq of quaternion of rotation at kth (current) sample
#             P_kp1_kp1     : State covariance matrix at kth (current) sample
#             b_hat_kp1_kp1 : Gyro noise bias at kth sample (current) sample
#         A namedtuple is defined to distinguish easily between the components being returned.
#     """

#     return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1 K_kp1")
#     return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
#                                    P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None, K_kp1 = None)
    
#     gravity_vector = init_gravity_vector # Gravity in the +x direction
#                                          # as the sensors are oriented such that +x aligns with g

#     # Assume that noise is equally distributed in the three spatial directions
#     # Noise covariance matrix becomes and identity matrix
#     Nr = (sts.sigma_omega_r_c**2)*geo.eye3 # Noise covariance for angular velocity noise
#     Nw = (sts.sigma_omega_w_c**2)*geo.eye3 # Noise covariance matrix for angular velocity bias rate noise
#     R  = (sts.sigma_accel_m_c**2)*geo.eye3 # Noise covariance matrix for acceleration measurement
#     Qc = np.block([[Nr,geo.z3],[geo.z3,Nw]])

#     b_hat_kp1_k     = b_hat_k_k # Propagate the bias from previous step
#     omega_hat_kp1_k = omega_m_kp1 - b_hat_kp1_k

#     # Propagate quaternion using the quaternion from previous step q_hat_k_k
#     OMEGA = geo.cross_prod_mat_4

#     # Zeroth order integrator
#     q_hat_kp1_k = (geo.mat_exp(0.5*OMEGA(omega_hat_k_k)*dt)) @ q_hat_k_k

#     # First order integrator
#     # omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
#     # q_hat_kp1_k = geo.mat_exp(0.5*OMEGA(omega_bar)*dt) @ q_hat_k_k
#     # q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k

#     omega0 = omega_hat_kp1_k

#     # System matrix for error state space (continuous-time) (constant)
#     Fc = np.block([[-geo.cross_prod_mat_3(omega_hat_k_k), -geo.eye3], 
#                    [ geo.z3,                               geo.z3  ]])
    
#     Gc = np.block([[-geo.eye3, geo.z3  ],
#                    [ geo.z3,   geo.eye3]])
    
#     # Approximation for Qd_k
#     # Q11 = (sigma_omega_r_c**2*dt*geo.eye3) + sigma_omega_w_c**2 * ((geo.eye3 * ((dt**3)/3)) + (((((norm(omega0) * dt)**3)/3) + 2*np.sin(norm(omega0)*dt) - (2*norm(omega0)*dt) )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**5)))
#     # Q22 = (sigma_omega_w_c**2)*dt*geo.eye3
#     # Q12 = -(sigma_omega_w_c**2) * ( (geo.eye3*((dt**2)/2)) - ((((norm(omega0)*dt) - np.sin(norm(omega0)*dt))/(norm(omega0))**3)*geo.cross_prod_mat_3(omega0)) + (((((norm(omega0) * dt)**2)/2) + np.cos(norm(omega0)*dt) - 1 )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**4)))
#     # Q21 = Q12.transpose()
#     # Qd_k_ = np.block([[Q11, Q12],
#     #                   [Q21, Q22]])

#     # State transition matrix for the discrete time error state space
#     # Phi_kp1 = geo.mat_exp(Fc*dt)
#     # Phi_kp1 = Fc*dt
#     # Phi_kp1 = Fc*dt + 0.5*(Fc*dt)@(Fc*dt)
#     Phi_kp1 = geo.mat_exp(Fc*dt) - geo.eye6 # TODO: Why eye6?

#     # Compute discrete-time process noise covariance matrix
#     Qd_k = Phi_kp1 @ Gc @ Qc @ Gc.transpose() @ Phi_kp1.transpose() * dt

#     # Test if the approximation is correct
#     # deltaQd = Qd_k - Qd_k_
#     # deltaQd[np.abs(deltaQd) < 1e-8] = 0
#     # assert(np.all(deltaQd == np.zeros(shape=(6,6))))

#     # Predicted state covariance matrix
#     P_kp1_k = Phi_kp1 @ P_k_k @ Phi_kp1.transpose() + Qd_k

#     # Kalman Update --------------------------------------------------------------------------------
#     # Given the propagated state estimates q_hat_kp1_k, b_hat_kp1_k, P_kp1_k, z_kp1 (current measurement), H (meas matrix)

#     # Method to accompany acceleration: failed
#     # R_q_hat_k_k = geo.quaternion_to_rotation_matrix(q_hat_k_k)
#     # R_q_hat_kp1_k = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
#     # a_hat_world = R_q_hat_k_k @ acc_r + gravity_vector
#     # a_hat_m     = R_q_hat_kp1_k.T @ (a_hat_world - gravity_vector)
#     # z_kp1 = acc_r - a_hat_m
#     # H_kp1 = np.block([-R_q_hat_kp1_k.T @ geo.cross_prod_mat_3(a_hat_world - gravity_vector), geo.z3])
#     # r_kp1 = z_kp1
    
#     # Measurement matrix
#     C = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
#     H_kp1 = np.block([geo.cross_prod_mat_3(C.T @ gravity_vector), np.zeros(shape=(3,3))])
#     # H_kp1 = np.block([geo.cross_prod_mat_3(C.T @ gravity_vector), geo.z3]) # TEST

#     z_hat_kp1 = C.T @ gravity_vector

#     # Compute residual
#     z_kp1 = acc_r
#     r_kp1 = z_kp1 - z_hat_kp1

#     # Covariance of residual
#     S_kp1 = H_kp1 @ P_kp1_k @ H_kp1.transpose() + R

#     # Kalman gain
#     K_kp1 = P_kp1_k @ H_kp1.transpose() @ np.linalg.inv(S_kp1)

#     # Correction
#     delta_theta_hat_plus = K_kp1 @ r_kp1

#     delta_q_hat_plus = 0.5*delta_theta_hat_plus[0:3] # Error angle vector
#     delta_b_hat_plus = delta_theta_hat_plus[3:6] # Gyro noise bias delta

#     delta_q_plus_dot_prod = delta_q_hat_plus.transpose() @ delta_q_hat_plus

#     if delta_q_plus_dot_prod <= 1:
#         delta_quat_hat = np.block([[delta_q_hat_plus],[np.sqrt(1 - delta_q_plus_dot_prod)]])
#     else:
#         delta_quat_hat = (1/np.sqrt(1 + delta_q_plus_dot_prod))*np.block([[delta_q_hat_plus],[1]])
#     #endif

#     dq_plus = delta_quat_hat # Quaternion delta (correction to be applied)
#     # dq_plus = np.block([[0.5*delta_x_hat_plus[0:3]],[np.array([1])]]) # Quaternion delta (correction to be applied)

#     # Apply correction
#     q_kp1_kp1 = geo.quaternion_multiply((dq_plus)/(norm(dq_plus)), q_hat_kp1_k) # Quaternion update
#     b_hat_kp1_kp1 = b_hat_kp1_k + delta_b_hat_plus # Gyro noise bias update

#     # Calculate rotation matrix corresponding to q_kp1_kp1
#     C_kp1 = geo.quaternion_to_rotation_matrix(q_kp1_kp1)

#     # Compute updated state covariance matrix
#     P_kp1_kp1 = (geo.eye6-K_kp1@H_kp1) @ P_kp1_k @ (geo.eye6-K_kp1@H_kp1).transpose() + K_kp1 @ R @ K_kp1.transpose()

#     # Update the estimated angular velocity using new estimation for bias
#     omega_hat_kp1_kp1 = omega_m_kp1 - b_hat_kp1_kp1

#     # Populate return list
#     return_list = return_list_type(C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1, K_kp1)

#     return return_list
# #enddef

def ukf_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k, init_gravity_vector):
    """
    Applies a step of Unscented Kalman filtering

    Parameters
    ---------------
    P_k_k : 6x6 matrix
        State covariance matrix of (k-1)th (previous) step
    dt : float
        Sampling interval
    omega_m_kp1 : vector
        Angular velocity vector
    acc_r : vector
        Acceleration vector
    q_hat_k_k : quaternion
        Quaternion of rotation at (k-1)th (previous) step
    b_hat_k_k : vector
        Gyro noise bias at (k-1)th (previous) step
    sigma_r : float
        Standard deviation of angular velocity noise
    sigma_b : float
        Standard deviation of gyro noise bias
    omega_hat_k_k : vector
        Angular velocity vector of the previous step
    init_gravity_vector : vector
        Initial gravity vector, used to align the sensors with the direction of gravity

    Returns
    -------
    namedtuple
        A namedtuple object containing:
            C_kp1         : Rotation matrix at kth (current) sample
            q_kp1_kp1     : Quaternion of rotation at kth (current) sample
            dq_plus       : dq of quaternion of rotation at kth (current) sample
            P_kp1_kp1     : State covariance matrix at kth (current) sample
            b_hat_kp1_kp1 : Gyro noise bias at kth sample (current) sample
            K_kp1         : Kalman gain at kth (current) sample
    """

    return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1 K_kp1")
    return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
                                   P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None, K_kp1 = None)
    
    gravity_vector = init_gravity_vector 

    # --- CRITICAL FIX: Use the passed arguments for noise ---
    # Use sigma_r (Dynamic/Static trust) passed from the main script
    Nr = (sigma_r**2) * geo.eye3 
    
    # Use sigma_b (Bias noise) passed from the main script
    Nb = (sigma_b**2) * geo.eye3 
    
    # Keep Acceleration noise fixed from settings (or add a parameter for it if needed)
    R  = (sts.sigma_accel_m_c**2) * geo.eye3 

    n     = 6    
    kappa = 0    
    alpha = 1e-3 
    beta  = 2    
    

    # return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1 K_kp1")
    # return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
    #                                 P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None, K_kp1 = None)
    
    # gravity_vector = init_gravity_vector # Gravity in the +x direction
    #                                       # as the sensors are oriented such that +x aligns with g

    # # Assume that noise is equally distributed in the three spatial directions
    # # Noise covariance matrix becomes and identity matrix
    # Nr = (sigma_omega_r_c**2)*geo.eye3 # Noise covariance for angular velocity noise
    # Nb = (sigma_omega_w_c**2)*geo.eye3 # Noise covariance matrix for angular velocity bias noise
    # R  = (sigma_accel_m_c**2)*geo.eye3 # Noise covariance matrix for acceleration measurement

    # n     = 6    # State dimension (quaternion axis + bias)
    # kappa = 0    # Secondary (default) scaling parameter
    # alpha = 1e-3 # Spread of sigma points
    # beta  = 2    # beta=2 for Gaussian distributions (incorporates prior knowledge of the distribution)
    
    lambda_ = alpha**2 * (n + kappa) - n # Scaling parameter for sigma points
    gamma   = np.sqrt(n + lambda_)
    
     # Initial state and covariance
    X = np.concatenate((omega_hat_k_k, b_hat_k_k)) # Augmented state vector (size 6)
    P_aug = P_k_k  # State covariance matrix (size 6x6)
    
    # Regularization to ensure positive definiteness
    epsilon = 1e-3  # Regularization factor
    P_aug += epsilon * np.eye(P_aug.shape[0])
    
    # Check eigenvalues for positive definiteness
    if np.any(np.linalg.eigvals(P_aug) <= 0):
        raise ValueError('Covariance matrix is not positive definite.')
    #endif

    # Calculate sigma points
    sqrtP = np.linalg.cholesky((n + lambda_) * P_aug)
    sqrtP = np.expand_dims(sqrtP, axis=1)

    sigma_points = np.zeros((n,1, 2 * n + 1))
    
    sigma_points[:,:, 0] = X # Mean sigma point

    for i in range(n):
        sigma_points[:,:,i + 1]     = X + gamma * sqrtP[:,:, i]
        sigma_points[:,:,i + n + 1] = X - gamma * sqrtP[:,:, i]
    #endfor

    # Propagate sigma points through the process model
    X_pred = np.zeros((n,1, 2 * n + 1))
    for i in range(2 * n + 1):
        omega_hat_kp1_k = omega_m_kp1 - sigma_points[3:6,:, i]

        # Propagate quaternion using the quaternion from previous step q_hat_k_k
        OMEGA = geo.cross_prod_mat_4
        
        # Zeroth order integrator
        # q_hat_kp1_k = (geo.mat_exp(0.5*OMEGA(omega_hat_k_k)*dt)) @ q_hat_k_k

        # First order integrator
        omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
        q_hat_kp1_k = geo.mat_exp(OMEGA(0.5*dt*omega_bar)) @ q_hat_k_k
        q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k
        
        X_pred[0:3,:, i] = omega_bar
        X_pred[3:6,:, i] = sigma_points[3:6, :,i] # Bias remains unchanged
    #endfor

    # Predicted state mean
    Wm = np.hstack((lambda_ / (n + lambda_), np.full(2 * n, 0.5 / (n + lambda_))))
    X_mean = np.dot(X_pred, Wm)
    
    # Predicted covariance
    P_kp1_k = np.zeros((n, n))
    for i in range(2 * n + 1):
        delta_theta_hat_plus = X_pred[:, :,i] - X_mean
        P_kp1_k += Wm[i] * np.outer(delta_theta_hat_plus, delta_theta_hat_plus)
    #endfor

    P_kp1_k += np.block([[Nr, np.zeros((3, 3))], [np.zeros((3, 3)), Nb]])  # Process noise
    
    # # Compute discrete-time process noise covariance matrix
    # omega0 = omega_hat_kp1_k
    # Q11 = (sigma_omega_r_c**2*dt*geo.eye3) + sigma_omega_w_c**2 * ((geo.eye3 * ((dt**3)/3)) + (((((norm(omega0) * dt)**3)/3) + 2*np.sin(norm(omega0)*dt) - (2*norm(omega0)*dt) )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**5)))
    # Q22 = (sigma_omega_w_c**2)*dt*geo.eye3
    # Q12 = (sigma_omega_w_c**2) * ( (geo.eye3*((dt**2)/2)) - ((((norm(omega0)*dt) - np.sin(norm(omega0)*dt))/(norm(omega0))**3)*geo.cross_prod_mat_3(omega0)) + (((((norm(omega0) * dt)**2)/2) + np.cos(norm(omega0)*dt) - 1 )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**4)))
    # Q21 = Q12.transpose()
    # Qd_k = np.block([[Q11, Q12],
    #                  [Q21, Q22]])

    # # System matrix for error state space (continuous-time) (constant)
    # Fc = np.block([[-geo.cross_prod_mat_3(omega_hat_k_k), -geo.eye3            ], 
    #                [np.zeros(shape=(3,3)),                np.zeros(shape=(3,3))]])
    
    # # State transition matrix for the discrete time error state space
    # Phi_kp1 = geo.mat_exp(Fc*dt)

    # # State covariance matrix
    # P_kp1_k = Phi_kp1 @ P_k_k @ Phi_kp1.transpose() + Qd_k

    # Kalman Update --------------------------------------------------------------------------------
    # Given the propagated state estimates q_hat_kp1_k, b_hat_kp1_k, P_kp1_k, z_kp1 (current measurement), H (meas matrix)
    
    # Measurement prediction
    z_hat_kp1 = np.zeros((3,1, 2 * n + 1))
    for i in range(2 * n + 1):
        C = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
        z_hat_kp1[:, :,i] = C @ gravity_vector
    #endfor

    z_mean = np.dot(z_hat_kp1, Wm)  # Predicted measurement mean
    
    # Measurement covariance and cross covariance
    Pzz = np.zeros((3, 3))
    Pxz = np.zeros((n, 3))

    for i in range(2 * n + 1):
        dZ = z_hat_kp1[:,:, i] - z_mean
        delta_theta_hat_plus = X_pred[:,:, i] - X_mean
        Pzz += Wm[i] * np.outer(dZ, dZ)
        Pxz += Wm[i] * np.outer(delta_theta_hat_plus, dZ)
    #endfor

    Pzz += R  # Add measurement noise
    
    # Kalman gain
    K_kp1 = Pxz @ np.linalg.inv(Pzz)

    # Compute residual
    z_kp1 = acc_r
    r_kp1 = z_kp1 - z_mean
    
    # Correction
    delta_theta_hat_plus = K_kp1 @ r_kp1
    
    dq_plus = np.concatenate((0.5 * delta_theta_hat_plus[0:3,0], [1]))
    
    dq_plus = np.expand_dims(dq_plus, axis=1)
    q_kp1_kp1 = geo.quaternion_multiply((dq_plus)/(norm(dq_plus)), q_hat_kp1_k) # Quaternion update
    b_hat_kp1_kp1 = X_mean[3:6] + delta_theta_hat_plus[3:6] # Gyro noise bias update
    
    # Calculate rotation matrix corresponding to q_kp1_kp1
    C_kp1 = geo.quaternion_to_rotation_matrix(q_kp1_kp1)

    # Compute updated covariance matrix
    P_kp1_kp1 = P_kp1_k - np.dot(K_kp1, np.dot(Pzz, K_kp1.T))

    # Update the estimated angular velocity using new estimation for bias
    omega_hat_kp1_kp1 = X_mean[0:3] + delta_theta_hat_plus[0:3]

    # Populate return list
    return_list = return_list_type(C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1, K_kp1)

    return return_list
#enddef

acc_r_corr = geo.l2v([0,0,0])

def test_step(P_k_k, dt, omega_m_kp1, acc_r, q_hat_k_k, b_hat_k_k, sigma_r, sigma_b, omega_hat_k_k, init_gravity_vector):
    global acc_r_corr
    """
    Applies a step of Indirect Kalman filtering

    Parameters
    ---------------
    P_k_k : 6x6 matrix
        State covariance matrix of (k-1)th (previous) step
    dt : float
        Sampling interval
    omega_m_kp1 : vector
        Angular velocity vector
    acc_r : vector
        Acceleration vector
    q_hat_k_k : quaternion
        Quaternion of rotation at (k-1)th (previous) step
    b_hat_k_k : vector
        Gyro noise bias at (k-1)th (previous) step
    sigma_r : float
        Standard deviation of angular velocity noise
    sigma_b : float
        Standard deviation of gyro noise bias
    omega_hat_k_k : vector
        Angular velocity vector of the previous step
    init_gravity_vector : vector
        Initial gravity vector, used to align the sensors with the direction of gravity
    
    Returns
    -------
    namedtuple
        A namedtuple object containing:
            C_kp1         : Rotation matrix at kth (current) sample
            q_kp1_kp1     : Quaternion of rotation at kth (current) sample
            dq_plus       : dq of quaternion of rotation at kth (current) sample
            P_kp1_kp1     : State covariance matrix at kth (current) sample
            b_hat_kp1_kp1 : Gyro noise bias at kth sample (current) sample
        A namedtuple is defined to distinguish easily between the components being returned.
    """

    return_list_type = namedtuple("return_list_type", "C_kp1 q_kp1_kp1 dq_plus P_kp1_kp1 b_hat_kp1_kp1 omega_kp1_kp1 K_kp1")
    return_list = return_list_type(C_kp1 = None, q_kp1_kp1 = None, dq_plus = None, 
                                   P_kp1_kp1 = None, b_hat_kp1_kp1 = None, omega_kp1_kp1 = None, K_kp1 = None)
    
    gravity_vector = init_gravity_vector # Gravity in the +x direction
                                         # as the sensors are oriented such that +x aligns with g

    # Assume that noise is equally distributed in the three spatial directions
    # Noise covariance matrix becomes and identity matrix
    Nr = (sts.sigma_omega_r_c**2)*geo.eye3 # Noise covariance for angular velocity noise
    Nw = (sts.sigma_omega_w_c**2)*geo.eye3 # Noise covariance matrix for angular velocity bias rate noise
    R  = (sts.sigma_accel_m_c**2)*geo.eye3 # Noise covariance matrix for acceleration measurement
    Qc = np.block([[Nr, geo.z3],[geo.z3, Nw]])

    b_hat_kp1_k     = b_hat_k_k # Propagate the bias from previous step
    omega_hat_kp1_k = omega_m_kp1 - b_hat_kp1_k

    # Propagate quaternion using the quaternion from previous step q_hat_k_k
    OMEGA = geo.cross_prod_mat_4

    # Zeroth order integrator
    q_hat_kp1_k = (geo.mat_exp(0.5*OMEGA(omega_hat_k_k)*dt)) @ q_hat_k_k

    # First order integrator
    # omega_bar   = 0.5*(omega_hat_k_k + omega_hat_kp1_k)
    # q_hat_kp1_k = geo.mat_exp(0.5*OMEGA(omega_bar)*dt) @ q_hat_k_k
    # q_hat_kp1_k += ((dt**2)/48.0)*(OMEGA(omega_hat_kp1_k)@OMEGA(omega_hat_k_k) - OMEGA(omega_hat_k_k)@OMEGA(omega_hat_kp1_k)) @ q_hat_k_k

    omega0 = omega_hat_kp1_k

    # System matrix for error state space (continuous-time) (constant)
    Fc = np.block([[-geo.cross_prod_mat_3(omega_hat_k_k), -geo.eye3], 
                   [ geo.z3,                               geo.z3  ]])
    
    Gc = np.block([[-geo.eye3, geo.z3  ],
                   [ geo.z3,   geo.eye3]])
    
    # Approximation for Qd_k
    # Q11 = (sigma_omega_r_c**2*dt*geo.eye3) + sigma_omega_w_c**2 * ((geo.eye3 * ((dt**3)/3)) + (((((norm(omega0) * dt)**3)/3) + 2*np.sin(norm(omega0)*dt) - (2*norm(omega0)*dt) )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**5)))
    # Q22 = (sigma_omega_w_c**2)*dt*geo.eye3
    # Q12 = -(sigma_omega_w_c**2) * ( (geo.eye3*((dt**2)/2)) - ((((norm(omega0)*dt) - np.sin(norm(omega0)*dt))/(norm(omega0))**3)*geo.cross_prod_mat_3(omega0)) + (((((norm(omega0) * dt)**2)/2) + np.cos(norm(omega0)*dt) - 1 )*(geo.sqrm(geo.cross_prod_mat_3(omega0))/norm(omega0)**4)))
    # Q21 = Q12.transpose()
    # Qd_k_ = np.block([[Q11, Q12],
    #                   [Q21, Q22]])

    # State transition matrix for the discrete time error state space
    # Phi_kp1 = geo.mat_exp(Fc*dt)
    # Phi_kp1 = Fc*dt
    # Phi_kp1 = Fc*dt + 0.5*(Fc*dt)@(Fc*dt)
    Phi_kp1 = geo.mat_exp(Fc*dt) - geo.eye6 # TODO: Why eye6?

    # Compute discrete-time process noise covariance matrix
    Qd_k = Phi_kp1 @ Gc @ Qc @ Gc.transpose() @ Phi_kp1.transpose() * dt

    # Test if the approximation is correct
    # deltaQd = Qd_k - Qd_k_
    # deltaQd[np.abs(deltaQd) < 1e-8] = 0
    # assert(np.all(deltaQd == np.zeros(shape=(6,6))))

    # Predicted state covariance matrix
    P_kp1_k = Phi_kp1 @ P_k_k @ Phi_kp1.transpose() + Qd_k

    # Kalman Update --------------------------------------------------------------------------------
    # Given the propagated state estimates q_hat_kp1_k, b_hat_kp1_k, P_kp1_k, z_kp1 (current measurement), H (meas matrix)

    # Method to accompany acceleration: failed
    # R_q_hat_k_k = geo.quaternion_to_rotation_matrix(q_hat_k_k)
    # R_q_hat_kp1_k = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
    # a_hat_world = R_q_hat_k_k @ acc_r + gravity_vector
    # a_hat_m     = R_q_hat_kp1_k.T @ (a_hat_world - gravity_vector)
    # z_kp1 = acc_r - a_hat_m
    # H_kp1 = np.block([-R_q_hat_kp1_k.T @ geo.cross_prod_mat_3(a_hat_world - gravity_vector), geo.z3])
    # r_kp1 = z_kp1
    
    # Measurement matrix
    C = geo.quaternion_to_rotation_matrix(q_hat_kp1_k)
    H_kp1 = np.block([geo.cross_prod_mat_3(C.T @ gravity_vector), np.zeros(shape=(3,3))])
    # H_kp1 = np.block([geo.cross_prod_mat_3(C.T @ gravity_vector), geo.z3]) # TEST

    z_hat_kp1 = C.T @ gravity_vector

    # Compute residual
    z_kp1 = acc_r
    r_kp1 = z_kp1 - z_hat_kp1

    # Covariance of residual
    S_kp1 = H_kp1 @ P_kp1_k @ H_kp1.transpose() + R

    # Kalman gain
    K_kp1 = P_kp1_k @ H_kp1.transpose() @ np.linalg.inv(S_kp1)

    # if (np.abs(np.linalg.norm(acc_r)-np.linalg.norm(sts.g_earth_vector)) < 0.1e-1):
    if np.abs(geo.angle_between(acc_r, z_hat_kp1)) < 0.1 and (np.abs(np.linalg.norm(acc_r)-np.linalg.norm(sts.g_earth_vector)) < 0.5e-1):
        acc_r = acc_r
        rot_gravity = C.T @ geo.l2v(sts.g_earth_vector)

        acc_r_corr = (acc_r - rot_gravity)

        #region test

        # Compute initial orientation quaternion (from body to earth)
        accel_body = rot_gravity
        corr_axis = np.cross(accel_body.flatten(), sts.g_earth_vector)
        corr_norm_axis = np.linalg.norm(corr_axis)
        if corr_norm_axis < 1e-8:
            corr_axis = np.array([1.0, 0.0, 0.0]) # If the axis is zero, pick a default orthogonal axis (e.g., [1, 0, 0])
            corr_norm_axis = 1.0
        #endif
        corr_axis = corr_axis / corr_norm_axis
        corr_axis = np.reshape(corr_axis, shape=(3,1))

        corr_angle = np.arccos(np.dot(accel_body.flatten(), sts.g_earth_vector) / (np.linalg.norm(accel_body) * np.linalg.norm(sts.g_earth_vector)))

        # Initial orientation quaternion (from body to earth)
        q_corr = geo.axis_angle_to_quaternion(corr_axis, corr_angle)

        q_hat_kp1_k = (q_corr + q_hat_kp1_k)/2

        #endregion test
    #endif

    # Correction
    delta_theta_hat_plus = K_kp1 @ r_kp1

    delta_q_hat_plus = 0.5*delta_theta_hat_plus[0:3] # Error angle vector
    delta_b_hat_plus = delta_theta_hat_plus[3:6] # Gyro noise bias delta

    delta_q_plus_dot_prod = delta_q_hat_plus.transpose() @ delta_q_hat_plus

    if delta_q_plus_dot_prod <= 1:
        delta_quat_hat = np.block([[delta_q_hat_plus],[np.sqrt(1 - delta_q_plus_dot_prod)]])
    else:
        delta_quat_hat = (1/np.sqrt(1 + delta_q_plus_dot_prod))*np.block([[delta_q_hat_plus],[1]])
    #endif

    dq_plus = delta_quat_hat # Quaternion delta (correction to be applied)
    # dq_plus = np.block([[0.5*delta_x_hat_plus[0:3]],[np.array([1])]]) # Quaternion delta (correction to be applied)

    # Apply correction
    q_kp1_kp1 = geo.quaternion_multiply((dq_plus)/(norm(dq_plus)), q_hat_kp1_k) # Quaternion update
    b_hat_kp1_kp1 = b_hat_kp1_k + delta_b_hat_plus # Gyro noise bias update

    # Calculate rotation matrix corresponding to q_kp1_kp1
    C_kp1 = geo.quaternion_to_rotation_matrix(q_kp1_kp1)

    # Compute updated state covariance matrix
    P_kp1_kp1 = (geo.eye6-K_kp1@H_kp1) @ P_kp1_k @ (geo.eye6-K_kp1@H_kp1).transpose() + K_kp1 @ R @ K_kp1.transpose()

    # Update the estimated angular velocity using new estimation for bias
    omega_hat_kp1_kp1 = omega_m_kp1 - b_hat_kp1_kp1

    # Populate return list
    return_list = return_list_type(C_kp1, q_kp1_kp1, dq_plus, P_kp1_kp1, b_hat_kp1_kp1, omega_hat_kp1_kp1, K_kp1)

    return return_list
#enddef