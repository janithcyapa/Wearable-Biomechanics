"""
JAE.py  —  Joint Angle Estimator
=================================
Initiated : 2026-08-02

Takes raw IMU data from two devices (proximal and distal segments of a joint)
and returns:
  - Relative quaternion  q_rel = q_prox* ⊗ q_dist   (scipy [x,y,z,w] convention)
  - Euler angles (degrees) from q_rel in a chosen sequence
  - Swing-only relative quaternion (yaw/axial rotation stripped out)
  - Swing Euler angles (degrees)  ← the clean flexion/extension angle lives here

Key design decisions
---------------------
1. SHARED WORLD FRAME  — Both IKF instances are initialised from the same averaged
   gravity vector measured during a known static posture (calibration window).
   Without this, q_rel is a meaningless mix of two independent frames.

2. ACCELEROMETER GATING — During high-dynamic phases (explosive lifts, impacts),
   the raw accelerometer reads real body acceleration on top of gravity, which
   would corrupt the tilt correction. The gate suppresses the measurement update
   when the accelerometer norm deviates too far from 1 g.

3. SWING-TWIST DECOMPOSITION — Yaw (rotation about the shared gravity axis) drifts
   freely with no magnetometer and independently on each sensor. For a hinge joint
   (knee, elbow) only the swing component (flexion/extension) is meaningful.
   Decomposing q_rel into swing * twist and discarding the twist removes the
   accumulating yaw error from the joint angle estimate.

Parameters you must supply
---------------------------
acc_prox, gyr_prox : Nx3 arrays  — proximal IMU (thigh, upper arm, …)
acc_dist, gyr_dist : Nx3 arrays  — distal  IMU (shank, forearm, …)
dt                 : float        — sampling interval [seconds]
n_cal_samples      : int          — number of samples at the start of the trial
                                    during which the athlete holds a known static
                                    posture. Both sensors must be stationary.
                                    Default: 100 (1 s at 100 Hz, 0.5 s at 200 Hz)
acc_gate_threshold : float        — how far ||a|| may deviate from 1 g before the
                                    accelerometer update is suppressed.
                                    Default: 0.3  (dimensionless, same units as
                                    your accel data — must be in g, not m/s²)
euler_seq          : str          — Euler angle sequence for decomposing q_rel.
                                    Default: 'xyz'  (roll=x, pitch=y, yaw=z)
                                    For a sagittal-plane knee: pitch (index 1)
                                    is flexion/extension.
twist_axis_world   : array (3,)   — the axis to strip when doing swing-twist.
                                    Default: None → auto-set from the averaged
                                    gravity direction (vertical axis, i.e. yaw).
                                    Override if your joint's axial-rotation axis
                                    is not vertical.
ikf_sigma_r        : float        — gyro noise std dev [rad/s/√Hz]
ikf_sigma_b        : float        — gyro bias noise std dev [rad/s/√Hz]
ikf_sigma_a        : float        — accel noise std dev [m/s²/√Hz]  (in g units
                                    if your input is in g)

Output  (JAEResult namedtuple)
-------------------------------
q_rel          : Nx4    relative quaternion [x,y,z,w] at each timestep
euler_rel      : Nx3    Euler angles [deg] of q_rel in euler_seq order
q_swing        : Nx4    yaw-stripped relative quaternion [x,y,z,w]
euler_swing    : Nx3    Euler angles [deg] of q_swing  ← use this for flex angle
q_prox         : Nx4    absolute quaternion of proximal sensor [x,y,z,w]
q_dist         : Nx4    absolute quaternion of distal  sensor  [x,y,z,w]
acc_gated      : N      boolean mask — True where acc update was suppressed
g_shared       : (3,)   the shared gravity vector used for initialisation
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from collections import namedtuple
from typing import Optional

from ikf import IKF   # same directory


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
JAEResult = namedtuple(
    "JAEResult",
    [
        "q_rel",        # Nx4  relative quaternion [x,y,z,w]
        "euler_rel",    # Nx3  Euler angles of q_rel [deg]
        "q_swing",      # Nx4  yaw-free relative quaternion [x,y,z,w]
        "euler_swing",  # Nx3  Euler angles of q_swing [deg]
        "q_prox",       # Nx4  proximal absolute quaternion [x,y,z,w]
        "q_dist",       # Nx4  distal absolute quaternion [x,y,z,w]
        "acc_gated",    # N    bool — True where acc update was suppressed
        "g_shared",     # (3,) shared gravity reference vector
    ],
)


# ---------------------------------------------------------------------------
# Swing-twist decomposition
# ---------------------------------------------------------------------------
def _swing_twist(q_scipy: Rot, twist_axis: np.ndarray) -> tuple[Rot, Rot]:
    """
    Decompose a rotation q into swing * twist, where twist is a rotation
    purely around twist_axis.

    Returns (swing, twist) as scipy Rotation objects.

    The decomposition satisfies:  q = swing * twist
    Swing is the component that rotates without any component around twist_axis.
    """
    twist_axis = np.asarray(twist_axis, dtype=float)
    twist_axis /= np.linalg.norm(twist_axis)

    qv = q_scipy.as_quat()          # [x, y, z, w]
    vec = qv[:3]                     # vector part
    scl = qv[3]                      # scalar part  (w)

    # Project vector part onto twist axis
    proj = np.dot(vec, twist_axis) * twist_axis

    # Reconstruct twist quaternion  [proj_x, proj_y, proj_z, w]
    q_twist_raw = np.concatenate([proj, [scl]])
    n = np.linalg.norm(q_twist_raw)
    if n < 1e-10:
        # Degenerate case: rotation is pure swing — no twist
        twist = Rot.identity()
    else:
        twist = Rot.from_quat(q_twist_raw / n)

    # Swing = q * twist^{-1}
    swing = q_scipy * twist.inv()

    return swing, twist


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------
def estimate_joint_angles(
    acc_prox:           np.ndarray,
    gyr_prox:           np.ndarray,
    acc_dist:           np.ndarray,
    gyr_dist:           np.ndarray,
    dt:                 float,
    n_cal_samples:      int   = 100,
    acc_gate_threshold: float = 0.3,
    euler_seq:          str   = "xyz",
    twist_axis_world:   Optional[np.ndarray] = None,
    ikf_sigma_r:        Optional[float] = None,
    ikf_sigma_b:        Optional[float] = None,
    ikf_sigma_a:        Optional[float] = None,
) -> JAEResult:
    """
    Estimate the relative joint angle between two IMU-instrumented segments.

    Parameters
    ----------
    acc_prox : (N, 3) array
        Proximal segment accelerometer data.  Units: g  (divide by 9.81 if in m/s²)
    gyr_prox : (N, 3) array
        Proximal segment gyroscope data.  Units: rad/s
    acc_dist : (N, 3) array
        Distal segment accelerometer data.  Units: g
    gyr_dist : (N, 3) array
        Distal segment gyroscope data.  Units: rad/s
    dt : float
        Sampling interval in seconds (e.g. 0.01 for 100 Hz).
    n_cal_samples : int
        Number of initial static samples used to build the shared gravity reference.
        The subject must be still during this window.
    acc_gate_threshold : float
        Suppress the accelerometer update when |‖a‖ − 1| > threshold  (in g).
        0.3 is a reasonable default; lower it for slow tasks, raise it (≤0.5)
        if the gate triggers too aggressively.
    euler_seq : str
        Euler angle decomposition sequence for the output.  Default 'xyz'.
    twist_axis_world : (3,) array or None
        The axis (in the shared world frame) to treat as the "axial / yaw" axis
        that will be stripped in the swing-twist decomposition.
        None → use the shared gravity direction (vertical, i.e. yaw).
    ikf_sigma_r, ikf_sigma_b, ikf_sigma_a : float or None
        IKF noise parameters.  None → use IKF class defaults.

    Returns
    -------
    JAEResult namedtuple — see module docstring for field descriptions.
    """

    # ------------------------------------------------------------------
    # 0. Sanitise inputs
    # ------------------------------------------------------------------
    acc_prox = np.asarray(acc_prox, dtype=float)
    gyr_prox = np.asarray(gyr_prox, dtype=float)
    acc_dist = np.asarray(acc_dist, dtype=float)
    gyr_dist = np.asarray(gyr_dist, dtype=float)

    N = len(acc_prox)
    assert acc_prox.shape == (N, 3), f"acc_prox must be (N,3), got {acc_prox.shape}"
    assert gyr_prox.shape == (N, 3), f"gyr_prox must be (N,3), got {gyr_prox.shape}"
    assert acc_dist.shape == (N, 3), f"acc_dist must be (N,3), got {acc_dist.shape}"
    assert gyr_dist.shape == (N, 3), f"gyr_dist must be (N,3), got {gyr_dist.shape}"
    assert len(acc_dist) == N,       "All four arrays must have the same length"

    n_cal = min(n_cal_samples, N)

    # ------------------------------------------------------------------
    # 1. Build shared gravity reference from calibration window
    #    Use samples that are actually static (norm close to 1 g)
    # ------------------------------------------------------------------
    def _static_mean_gravity(acc: np.ndarray, n: int) -> np.ndarray:
        """Average accelerometer readings over the calibration window,
        excluding any sample that is clearly non-static."""
        cal = acc[:n]
        norms = np.linalg.norm(cal, axis=1)
        static_mask = np.abs(norms - 1.0) < 0.1   # within 0.1 g of 1 g
        if static_mask.sum() < 5:
            # Fallback: just use all cal samples
            static_mask = np.ones(n, dtype=bool)
        return cal[static_mask].mean(axis=0)

    g_prox = _static_mean_gravity(acc_prox, n_cal)
    g_dist = _static_mean_gravity(acc_dist, n_cal)

    # Average and normalise → shared world "down" direction
    g_shared = (g_prox + g_dist) / 2.0
    g_norm = np.linalg.norm(g_shared)
    if g_norm < 1e-8:
        raise ValueError(
            "Shared gravity vector is near-zero. Check that calibration "
            "samples are valid and the accelerometers are in g units."
        )
    g_shared /= g_norm

    # Twist axis: default to gravity direction (= yaw axis when gravity is [0,0,-1])
    if twist_axis_world is None:
        twist_axis_world = g_shared.copy()
    else:
        twist_axis_world = np.asarray(twist_axis_world, dtype=float)
        twist_axis_world /= np.linalg.norm(twist_axis_world)

    # ------------------------------------------------------------------
    # 2. Initialise both IKF filters with the SAME gravity reference
    # ------------------------------------------------------------------
    ikf_prox = IKF(dt=dt, sigma_r=ikf_sigma_r, sigma_b=ikf_sigma_b, sigma_a=ikf_sigma_a)
    ikf_dist = IKF(dt=dt, sigma_r=ikf_sigma_r, sigma_b=ikf_sigma_b, sigma_a=ikf_sigma_a)

    ikf_prox.reset(init_gravity_vector=g_shared)
    ikf_dist.reset(init_gravity_vector=g_shared)

    # ------------------------------------------------------------------
    # 3. Pre-allocate outputs
    # ------------------------------------------------------------------
    q_prox_out    = np.zeros((N, 4))   # [x,y,z,w]
    q_dist_out    = np.zeros((N, 4))
    q_rel_out     = np.zeros((N, 4))
    euler_rel_out = np.zeros((N, 3))
    q_swing_out   = np.zeros((N, 4))
    euler_sw_out  = np.zeros((N, 3))
    acc_gated_out = np.zeros(N, dtype=bool)

    # ------------------------------------------------------------------
    # 4. Step through the data
    # ------------------------------------------------------------------
    for i in range(N):
        ap = acc_prox[i]
        gp = gyr_prox[i]
        ad = acc_dist[i]
        gd = gyr_dist[i]

        # ---- Accelerometer gate (applied to both sensors independently) ----
        gate_p = bool(abs(np.linalg.norm(ap) - 1.0) > acc_gate_threshold
                      or not np.all(np.isfinite(ap)))
        gate_d = bool(abs(np.linalg.norm(ad) - 1.0) > acc_gate_threshold
                      or not np.all(np.isfinite(ad)))
        acc_gated_out[i] = gate_p or gate_d

        # When gated, pass the expected gravity so the IKF skips the
        # tilt correction naturally (residual ≈ 0 → K * r ≈ 0)
        ap_in = g_shared if gate_p else ap
        ad_in = g_shared if gate_d else ad

        # ---- Run both filters ----
        ikf_prox.step(ap_in, gp)
        ikf_dist.step(ad_in, gd)

        q_p = ikf_prox.q_hat.copy()   # [x,y,z,w]
        q_d = ikf_dist.q_hat.copy()

        q_prox_out[i] = q_p
        q_dist_out[i] = q_d

        # ---- Relative quaternion:  q_rel = q_prox* ⊗ q_dist ----
        r_prox = Rot.from_quat(q_p)
        r_dist = Rot.from_quat(q_d)
        r_rel  = r_prox.inv() * r_dist

        q_rel_out[i]     = r_rel.as_quat()
        euler_rel_out[i] = r_rel.as_euler(euler_seq, degrees=True)

        # ---- Swing-twist decomposition (strip yaw drift) ----
        swing, _ = _swing_twist(r_rel, twist_axis_world)
        q_swing_out[i]  = swing.as_quat()
        euler_sw_out[i] = swing.as_euler(euler_seq, degrees=True)

    # ------------------------------------------------------------------
    # 5. Pack and return
    # ------------------------------------------------------------------
    return JAEResult(
        q_rel       = q_rel_out,
        euler_rel   = euler_rel_out,
        q_swing     = q_swing_out,
        euler_swing = euler_sw_out,
        q_prox      = q_prox_out,
        q_dist      = q_dist_out,
        acc_gated   = acc_gated_out,
        g_shared    = g_shared,
    )


# ---------------------------------------------------------------------------
# Convenience: extract a single joint axis from the swing Euler output
# ---------------------------------------------------------------------------
def flexion_angle(result: JAEResult, axis_index: int = 1) -> np.ndarray:
    """
    Return the flexion/extension angle time series in degrees.

    Parameters
    ----------
    result : JAEResult
        Output from estimate_joint_angles().
    axis_index : int
        Which Euler component to treat as the flexion axis.
        For euler_seq='xyz':  0=roll(x), 1=pitch(y), 2=yaw(z)
        For a sagittal-plane knee with the IMU y-axis along the flexion axis: 1

    Returns
    -------
    angles : (N,) array of floats in degrees.
    """
    return result.euler_swing[:, axis_index]


# ---------------------------------------------------------------------------
# Example / smoke-test  (run this file directly to check it works)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    fs   = 100.0          # Hz
    dt   = 1.0 / fs
    T    = 10.0           # seconds
    N    = int(T * fs)
    t    = np.arange(N) * dt

    # --- Simulate a knee flexion: sinusoidal pitch rotation on distal sensor ---
    # Proximal (thigh): stays roughly still
    # Distal (shank): oscillates ±45° in pitch at 0.5 Hz

    flex_angle_true = 45.0 * np.sin(2 * np.pi * 0.5 * t)   # degrees
    flex_rad        = np.deg2rad(flex_angle_true)

    # Build synthetic gyro for distal: d(angle)/dt
    omega_flex = np.gradient(flex_rad, dt)   # rad/s

    # Proximal gyro: zero (stationary thigh)
    gyr_prox_sim = np.zeros((N, 3))
    gyr_dist_sim = np.column_stack([
        np.zeros(N),
        omega_flex,           # pitch rotation
        np.zeros(N),
    ])

    # Add small gyro bias and noise
    gyr_prox_sim += np.random.randn(N, 3) * np.radians(0.05)
    gyr_dist_sim += np.random.randn(N, 3) * np.radians(0.05)
    gyr_dist_sim += np.radians(0.5) / 3600   # small bias

    # Accelerometers: gravity = [0, 0, 1] in body frame (sensor z-axis points up)
    acc_prox_sim = np.tile([0.0, 0.0, 1.0], (N, 1))
    acc_dist_sim = np.tile([0.0, 0.0, 1.0], (N, 1))

    # Add small accel noise
    acc_prox_sim += np.random.randn(N, 3) * 1e-3
    acc_dist_sim += np.random.randn(N, 3) * 1e-3

    # --- Run JAE ---
    result = estimate_joint_angles(
        acc_prox           = acc_prox_sim,
        gyr_prox           = gyr_prox_sim,
        acc_dist           = acc_dist_sim,
        gyr_dist           = gyr_dist_sim,
        dt                 = dt,
        n_cal_samples      = 50,
        acc_gate_threshold = 0.3,
        euler_seq          = "xyz",
    )

    flex_est = flexion_angle(result, axis_index=1)   # pitch = flexion

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, flex_angle_true, label="True flexion angle", lw=2)
    axes[0].plot(t, flex_est, label="Estimated (swing pitch)", lw=1.5, ls="--")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Flexion/Extension Angle — True vs Estimated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, result.euler_rel[:, 1], label="q_rel pitch (with yaw drift)", alpha=0.7)
    axes[1].plot(t, flex_est, label="q_swing pitch (yaw stripped)", lw=1.5)
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title("Effect of Swing-Twist Decomposition")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, result.acc_gated.astype(float), color="tomato", label="Acc gate active")
    axes[2].set_ylabel("Gated")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Accelerometer Gate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nShared gravity vector:", result.g_shared)
    print(f"Peak flexion estimated: {flex_est.max():.2f} deg  (true: {flex_angle_true.max():.2f} deg)")
    print(f"Acc gate triggered: {result.acc_gated.sum()} / {N} samples")
