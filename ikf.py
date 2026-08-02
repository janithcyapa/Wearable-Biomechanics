import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R


class IKF:
    """
    Indirect Kalman Filter for IMU orientation estimation.
    This class is a clean, object-oriented implementation of the IKF algorithm,
    using standard libraries (numpy, scipy) for mathematical operations.
    """
    
    def __init__(self, dt, sigma_r=None, sigma_b=None, sigma_a=None):
        """
        Initialize the IKF filter.

        Parameters
        ----------
        dt : float
            Sampling interval in seconds.
        sigma_r : float, optional
            Standard deviation of angular velocity noise. Defaults to settings.
        sigma_b : float, optional
            Standard deviation of gyro noise bias. Defaults to settings.
        sigma_a : float, optional
            Standard deviation of acceleration measurement noise. Defaults to settings.
        """
        self.dt = dt
        
        # Defaults for measurement noises if not provided
        default_sigma_r = np.radians(0.007)
        default_sigma_b = np.radians(10) / 3600
        default_sigma_a = 180 * 9.81 * 1e-6
        
        self.sigma_r = sigma_r if sigma_r is not None else default_sigma_r
        self.sigma_b = sigma_b if sigma_b is not None else default_sigma_b
        self.sigma_a = sigma_a if sigma_a is not None else default_sigma_a
        
        self.q_hat = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        self.b_hat = np.zeros(3)
        self.P = np.eye(6) * 0.1
        self.omega_hat_prev = np.zeros(3)
        self.init_gravity_vector = None
        self.initialized = False

        self.R_cov = (self.sigma_a ** 2) * np.eye(3)
        self.Qc = np.block([
            [(self.sigma_r ** 2) * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), (self.sigma_b ** 2) * np.eye(3)]
        ])
        
    def _skew_symmetric(self, v):
        """Returns the 3x3 skew-symmetric matrix of a 3D vector."""
        v = np.asarray(v).ravel()
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])

    def reset(self, init_gravity_vector=None):
        """Resets the state of the filter."""
        self.q_hat = np.array([0.0, 0.0, 0.0, 1.0])
        self.b_hat = np.zeros(3)
        self.P = np.eye(6) * 0.1
        self.omega_hat_prev = np.zeros(3)
        
        if init_gravity_vector is not None:
            self.init_gravity_vector = np.asarray(init_gravity_vector).ravel()
            self.initialized = True
        else:
            self.init_gravity_vector = None
            self.initialized = False

    def step(self, acc_meas, gyr_meas):
        """
        Applies a single step of the IKF algorithm.

        Parameters
        ----------
        acc_meas : array-like
            Acceleration vector measurement.
        gyr_meas : array-like
            Angular velocity vector measurement.

        Returns
        -------
        C_kp1 : ndarray
            The 3x3 rotation matrix for the current sample.
        """
        acc_meas = np.asarray(acc_meas).ravel()
        gyr_meas = np.asarray(gyr_meas).ravel()

        if not self.initialized:
            if np.linalg.norm(acc_meas) > 1e-8 and np.all(np.isfinite(acc_meas)):
                self.init_gravity_vector = acc_meas.copy()
            else:
                self.init_gravity_vector = np.array([0.0, 0.0, 1.0])
            self.initialized = True

        if np.linalg.norm(acc_meas) < 1e-8 or not np.all(np.isfinite(acc_meas)):
            acc_meas = self.init_gravity_vector

        # 1. Bias correction and Propagate quaternion
        r_old = R.from_quat(self.q_hat)
        r_delta = R.from_rotvec(self.omega_hat_prev * self.dt)
        r_new_pred = r_old * r_delta
        
        # 2. State transition matrix for error state space
        Fc = np.zeros((6, 6))
        Fc[0:3, 0:3] = -self._skew_symmetric(self.omega_hat_prev)
        Fc[0:3, 3:6] = -np.eye(3)

        Gc = np.zeros((6, 6))
        Gc[0:3, 0:3] = -np.eye(3)
        Gc[3:6, 3:6] = np.eye(3)

        Phi_kp1 = scipy.linalg.expm(Fc * self.dt)
        
        # 3. Process noise covariance
        Qd_k = Gc @ self.Qc @ Gc.T * self.dt

        # 4. Predicted state covariance
        P_kp1_k = Phi_kp1 @ self.P @ Phi_kp1.T + Qd_k

        # 5. Measurement update
        C_pred = r_new_pred.as_matrix()
        z_hat_kp1 = C_pred.T @ self.init_gravity_vector
        
        H_kp1 = np.zeros((3, 6))
        H_kp1[0:3, 0:3] = self._skew_symmetric(z_hat_kp1)
        
        r_kp1 = acc_meas - z_hat_kp1
        
        S_kp1 = H_kp1 @ P_kp1_k @ H_kp1.T + self.R_cov
        
        K_kp1 = P_kp1_k @ H_kp1.T @ np.linalg.inv(S_kp1)
        
        delta_theta_hat_plus = K_kp1 @ r_kp1
        
        delta_q_hat_plus = 0.5 * delta_theta_hat_plus[0:3]
        delta_b_hat_plus = delta_theta_hat_plus[3:6]
        
        # 6. Quaternion delta
        delta_q_plus_dot_prod = np.dot(delta_q_hat_plus, delta_q_hat_plus)
        if delta_q_plus_dot_prod <= 1:
            delta_quat_hat = np.concatenate([delta_q_hat_plus, [np.sqrt(1 - delta_q_plus_dot_prod)]])
        else:
            delta_quat_hat = np.concatenate([delta_q_hat_plus, [1.0]]) / np.sqrt(1 + delta_q_plus_dot_prod)

        dq_plus = delta_quat_hat / np.linalg.norm(delta_quat_hat)
        
        # 7. Apply correction
        r_dq_plus = R.from_quat(dq_plus)
        r_kp1_kp1 = r_new_pred * r_dq_plus
        
        self.q_hat = r_kp1_kp1.as_quat()
        self.b_hat = self.b_hat + delta_b_hat_plus
        
        # 8. Update covariance
        I_KH = np.eye(6) - K_kp1 @ H_kp1
        self.P = I_KH @ P_kp1_k @ I_KH.T + K_kp1 @ self.R_cov @ K_kp1.T
        
        # 9. Update the estimated angular velocity using new estimation for bias
        self.omega_hat_prev = gyr_meas - self.b_hat
        
        return r_kp1_kp1.as_matrix()

    def process(self, accel_data, gyro_data):
        """
        Process a complete data series and return rotation matrices.

        Parameters
        ----------
        accel_data : ndarray
            Nx3 array of acceleration vectors.
        gyro_data : ndarray
            Nx3 array of angular velocity vectors.

        Returns
        -------
        R_list : list of ndarray
            List of 3x3 rotation matrices for each time step.
        """
        accel_data = np.asarray(accel_data)
        gyro_data = np.asarray(gyro_data)
        N = len(accel_data)
        
        R_list = []
        for i in range(N):
            C = self.step(accel_data[i], gyro_data[i])
            R_list.append(C)
            
        return R_list
