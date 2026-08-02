import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import os

import kalman as kl
import settings as sts

# =============================================================================
# Right knee angle (Device 1 = proximal/thigh, Device 10 = distal/shank)
# Pipeline:
#   IKF orientation -> relative rotation -> Euler yaw (xyz, index=2)
#   unwrap -> (optional) drift removal (HP / SSA) -> range-shift to target angle range
# =============================================================================

# ----------------------- Pipeline parameters -----------------------
HP_ORDER = 2
HP_CUTOFF_HZ = 0.05

# Set this to False if you want to bypass the high-pass filter.
# True  -> final knee angle uses HPF + range shift
# False -> final knee angle uses RAW + range shift, with no HPF applied
USE_HPF = True

SSA_WINDOW_SECONDS = 4.0
SSA_TREND_RANK = 1

TARGET_MIN_DEG = 80.0
TARGET_MAX_DEG = 180.0

FORCE_CLIP_TO_RANGE = False
INIT_SAMPLES_DEFAULT = 50

DEBUG_BAD_ROTATIONS = False


# ----------------------- Joint setup (Right Knee only) -----------------------
RIGHT_KNEE_CFG = {
    "joint_name": "Right Knee",
    "proximal_prefix": "device4_IMU2",
    "distal_prefix": "device7_IMU1",

    # Extract yaw from relative rotation
    "euler_seq": "xyz",
    "euler_index": 2,

    # Flip if direction is wrong
    "flex_sign": 1.0,

    # Base mapping: angle = base_angle - bend
    "base_angle": 80.0,

    "target_min": TARGET_MIN_DEG,
    "target_max": TARGET_MAX_DEG,

    "init_samples": INIT_SAMPLES_DEFAULT,
}


# =============================================================================
# Helpers
# =============================================================================

def as_col(v, n):
    v = np.asarray(v)
    if v.shape == (n,):
        return v.reshape(n, 1)
    if v.shape == (n, 1):
        return v
    raise ValueError(f"Expected shape ({n},) or ({n},1), got {v.shape}")


def convert_physical_data(df, acc_cols, gyr_cols):
    missing = [c for c in (acc_cols + gyr_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    accel_data = df[acc_cols].values / sts.g
    gyro_data = np.deg2rad(df[gyr_cols].values)
    return accel_data, gyro_data


def _nearest_valid_init_gravity(accel_data, eps=1e-6, max_search=2000):
    N = len(accel_data)
    M = min(N, max_search)
    for i in range(M):
        a = np.asarray(accel_data[i], dtype=float).reshape(3, 1)
        if np.all(np.isfinite(a)) and (np.linalg.norm(a) > eps):
            return a, i
    return np.array([[0.0], [0.0], [1.0]]), 0


def _orthonormalize_R(C):
    U, _, Vt = np.linalg.svd(C)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def _is_valid_rotation(C, tol_ortho=1e-2, tol_det=1e-1):
    if C.shape != (3, 3):
        return False
    if not np.all(np.isfinite(C)):
        return False
    det = np.linalg.det(C)
    if not np.isfinite(det):
        return False
    if abs(det - 1.0) > tol_det:
        return False
    I = C.T @ C
    if np.linalg.norm(I - np.eye(3), ord="fro") > tol_ortho:
        return False
    return True


def run_adaptive_filter(accel_data, gyro_data, dt):
    q_current = np.array([[1.0], [0.0], [0.0], [0.0]])
    P_current = np.eye(6) * 0.1
    b_current = np.zeros((3, 1))
    omega_hat_prev = np.zeros((3, 1))

    init_gravity_vector, _ = _nearest_valid_init_gravity(accel_data)

    R_list = []
    C_prev = np.eye(3)

    for i in range(len(accel_data)):
        acc_meas = np.asarray(accel_data[i], dtype=float).reshape(3, 1)
        gyr_meas = np.asarray(gyro_data[i], dtype=float).reshape(3, 1)

        if (not np.all(np.isfinite(acc_meas))) or (np.linalg.norm(acc_meas) < 1e-8):
            acc_meas = init_gravity_vector

        q_current = as_col(q_current, 4)
        b_current = as_col(b_current, 3)
        omega_hat_prev = as_col(omega_hat_prev, 3)

        result = kl.ikf_step(
            P_k_k=P_current,
            dt=dt,
            omega_m_kp1=gyr_meas,
            acc_r=acc_meas,
            q_hat_k_k=q_current,
            b_hat_k_k=b_current,
            sigma_r=sts.sigma_omega_r_c,
            sigma_b=sts.sigma_omega_w_c,
            omega_hat_k_k=omega_hat_prev,
            init_gravity_vector=init_gravity_vector,
        )

        P_current = result.P_kp1_kp1
        q_current = as_col(result.q_kp1_kp1, 4)
        b_current = as_col(result.b_hat_kp1_kp1, 3)
        omega_hat_prev = as_col(result.omega_kp1_kp1, 3)

        C = np.asarray(result.C_kp1, dtype=float)

        if C.shape == (3, 3) and np.all(np.isfinite(C)):
            C = _orthonormalize_R(C)

        if not _is_valid_rotation(C):
            if DEBUG_BAD_ROTATIONS:
                det = np.linalg.det(C) if (C.shape == (3, 3) and np.all(np.isfinite(C))) else None
                print(f"[IKF] invalid C at i={i}, det={det}; using previous valid rotation.")
            C = C_prev
        else:
            C_prev = C

        R_list.append(C)

    return R_list


def rel_yaw_from_Crel(C_prox, C_dist, euler_seq="xyz", euler_index=2, flex_sign=1.0):
    """Return *wrapped* Euler component (yaw) in degrees from relative rotation."""
    C_prox = np.asarray(C_prox, dtype=float)
    C_dist = np.asarray(C_dist, dtype=float)

    if not _is_valid_rotation(C_prox):
        C_prox = _orthonormalize_R(C_prox) if (C_prox.shape == (3, 3) and np.all(np.isfinite(C_prox))) else np.eye(3)
    if not _is_valid_rotation(C_dist):
        C_dist = _orthonormalize_R(C_dist) if (C_dist.shape == (3, 3) and np.all(np.isfinite(C_dist))) else np.eye(3)

    r_p = R.from_matrix(C_prox)
    r_d = R.from_matrix(C_dist)
    r_rel = r_p.inv() * r_d

    ang_deg = r_rel.as_euler(euler_seq, degrees=True)[euler_index]
    ang_deg *= float(flex_sign)

    if ang_deg > 180.0:
        ang_deg -= 360.0
    elif ang_deg < -180.0:
        ang_deg += 360.0

    return float(ang_deg)


def unwrap_deg(x_deg, discont_deg=180.0):
    x_rad = np.deg2rad(np.asarray(x_deg, dtype=float))
    x_unwrapped_rad = np.unwrap(x_rad, discont=np.deg2rad(discont_deg))
    return np.rad2deg(x_unwrapped_rad)


def highpass_filter(x, fs, fc=HP_CUTOFF_HZ, order=HP_ORDER):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    wn = fc / (0.5 * fs)
    b, a = butter(order, wn, btype="highpass")
    return filtfilt(b, a, x)


def ssa_detrend(x, L=400, trend_rank=SSA_TREND_RANK):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if L < 2 or L >= N:
        raise ValueError("Choose L such that 2 <= L < len(x)")
    K = N - L + 1
    X = np.column_stack([x[i:i + K] for i in range(L)])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    X_trend = np.zeros_like(X)
    for r in range(min(trend_rank, len(S))):
        X_trend += S[r] * np.outer(U[:, r], Vt[r, :])

    def diag_avg(M):
        K_, L_ = M.shape
        N_ = K_ + L_ - 1
        y = np.zeros(N_)
        w = np.zeros(N_)
        for i in range(K_):
            for j in range(L_):
                y[i + j] += M[i, j]
                w[i + j] += 1
        return y / w

    x_trend = diag_avg(X_trend)
    return x - x_trend, x_trend


def shift_bend_to_force_angle_range(raw_bend, base_angle, target_min, target_max, force_clip=FORCE_CLIP_TO_RANGE):
    """
    Given: angle0 = base_angle - raw_bend
    Find constant shift s so that angle = angle0 - s fits within [target_min, target_max].
    """
    raw_bend = np.asarray(raw_bend, dtype=float)
    angle0 = base_angle - raw_bend

    a_min = float(np.min(angle0))
    a_max = float(np.max(angle0))

    s_low = a_max - target_max
    s_high = a_min - target_min

    feasible = (s_low <= s_high)
    s = 0.5 * (s_low + s_high)

    angle_shifted = angle0 - s
    if force_clip:
        angle_shifted = np.clip(angle_shifted, target_min, target_max)

    diag = {
        "angle0_min": a_min,
        "angle0_max": a_max,
        "s_low_needed": s_low,
        "s_high_needed": s_high,
        "feasible_shift_only": feasible,
        "s_used": float(s),
        "angle_shifted_min": float(np.min(angle_shifted)),
        "angle_shifted_max": float(np.max(angle_shifted)),
        "force_clip": bool(force_clip),
    }
    return angle_shifted, float(s), feasible, diag


# =============================================================================
# Column handling
# =============================================================================

def _prefix_variants(prefix: str):
    p = str(prefix)
    no_ = p.replace("_", "")
    return [p, p.lower(), p.upper(), no_, no_.lower(), no_.upper()]


def get_sensor_columns(df, prefix):
    for pv in _prefix_variants(prefix):
        acc_cols = [f"{pv}_{axis}" for axis in ["AX", "AY", "AZ"]]
        gyr_cols = [f"{pv}_{axis}" for axis in ["GX", "GY", "GZ"]]
        if all(c in df.columns for c in (acc_cols + gyr_cols)):
            return acc_cols, gyr_cols, pv
    raise ValueError(f"Could not find columns for prefix '{prefix}'. Tried: {_prefix_variants(prefix)}")


# =============================================================================
# Main knee pipeline
# =============================================================================

def estimate_right_knee(df, cfg, dt):
    print(f"\n--- Processing {cfg['joint_name']} ---")

    acc_p_cols, gyr_p_cols, p_used = get_sensor_columns(df, cfg["proximal_prefix"])
    acc_d_cols, gyr_d_cols, d_used = get_sensor_columns(df, cfg["distal_prefix"])
    print(f"Using prefixes: proximal='{p_used}', distal='{d_used}'")

    acc_p, gyr_p = convert_physical_data(df, acc_p_cols, gyr_p_cols)
    acc_d, gyr_d = convert_physical_data(df, acc_d_cols, gyr_d_cols)

    print("Running IKF for proximal IMU (Device 1)...")
    R_p_list = run_adaptive_filter(acc_p, gyr_p, dt)
    print("Running IKF for distal IMU (Device 10)...")
    R_d_list = run_adaptive_filter(acc_d, gyr_d, dt)

    if "Timestamp" in df.columns:
        t = df["Timestamp"].values
    else:
        t = np.arange(len(df)) * dt

    n = min(len(R_p_list), len(R_d_list), len(t))
    t = np.asarray(t[:n])

    # 1) Wrapped yaw series
    raw_yaw_wrapped = np.zeros(n, dtype=float)
    for i in range(n):
        raw_yaw_wrapped[i] = rel_yaw_from_Crel(
            R_p_list[i],
            R_d_list[i],
            euler_seq=cfg.get("euler_seq", "xyz"),
            euler_index=int(cfg.get("euler_index", 2)),
            flex_sign=float(cfg.get("flex_sign", 1.0)),
        )

    # 2) Unwrap across time
    raw_yaw = unwrap_deg(raw_yaw_wrapped, discont_deg=180.0)

    init_samples = int(min(cfg.get("init_samples", INIT_SAMPLES_DEFAULT), n))
    base_angle = float(cfg["base_angle"])

    # ---- RAW mapping (no drift removal) ----
    init_offset = float(np.mean(raw_yaw[:init_samples]))
    bend_raw = raw_yaw - init_offset
    angles_raw = base_angle - bend_raw  # raw (no detrending)
    angles_raw_rs, _, _, diag_raw = shift_bend_to_force_angle_range(
        raw_bend=bend_raw,
        base_angle=base_angle,
        target_min=float(cfg.get("target_min", TARGET_MIN_DEG)),
        target_max=float(cfg.get("target_max", TARGET_MAX_DEG)),
        force_clip=FORCE_CLIP_TO_RANGE,
    )

    fs = 1.0 / float(dt)

    # 3) Optional high-pass drift removal
    if USE_HPF:
        raw_yaw_hp = highpass_filter(raw_yaw, fs=fs, fc=HP_CUTOFF_HZ, order=HP_ORDER)
        init_hp = float(np.mean(raw_yaw_hp[:init_samples]))
        bend_hp = raw_yaw_hp - init_hp
        angles_hp, _, _, diag_hp = shift_bend_to_force_angle_range(
            raw_bend=bend_hp,
            base_angle=base_angle,
            target_min=float(cfg.get("target_min", TARGET_MIN_DEG)),
            target_max=float(cfg.get("target_max", TARGET_MAX_DEG)),
            force_clip=FORCE_CLIP_TO_RANGE,
        )
        angles_final = angles_hp
        final_method = "HPF + range shift"
    else:
        # No high-pass filtering. Use raw, offset-corrected, range-shifted angle as final result.
        raw_yaw_hp = np.full_like(raw_yaw, np.nan, dtype=float)
        angles_hp = np.full_like(angles_raw_rs, np.nan, dtype=float)
        diag_hp = {
            "hpf_used": False,
            "message": "HPF disabled. Final angle uses RAW + range shift.",
        }
        angles_final = angles_raw_rs
        final_method = "RAW + range shift; no HPF"

    # 4) SSA detrend drift removal (KL branch)
    L = int(fs * SSA_WINDOW_SECONDS)
    L = max(2, min(L, n - 1))
    raw_yaw_kl, trend = ssa_detrend(raw_yaw, L=L, trend_rank=SSA_TREND_RANK)

    init_kl = float(np.mean(raw_yaw_kl[:init_samples]))
    bend_kl = raw_yaw_kl - init_kl
    angles_kl, _, _, diag_kl = shift_bend_to_force_angle_range(
        raw_bend=bend_kl,
        base_angle=base_angle,
        target_min=float(cfg.get("target_min", TARGET_MIN_DEG)),
        target_max=float(cfg.get("target_max", TARGET_MAX_DEG)),
        force_clip=FORCE_CLIP_TO_RANGE,
    )

    info = {"raw": diag_raw, "hp": diag_hp, "kl": diag_kl, "final_method": final_method}

    return (
        t,
        angles_raw,     # raw (no range shift)
        angles_raw_rs,  # raw + range shift
        angles_hp,      # HPF + range shift; NaN if USE_HPF=False
        angles_final,   # selected final angle based on USE_HPF
        angles_kl,      # SSA detrended + range shift
        raw_yaw,
        raw_yaw_hp,
        raw_yaw_kl,
        trend,
        info,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    target_file = (
        "entry_0001_none_1781856275.csv"
    )

    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        raise SystemExit

    print(f"Loading dataset: {target_file}")
    df = pd.read_csv(target_file)

    dt = 0.01
    print(f"Using fixed time step (dt): {dt:.4f} s (fs={1.0/dt:.1f} Hz)")

    (
        t,
        angles_raw,
        angles_raw_rs,
        angles_hp,
        angles_final,
        angles_kl,
        raw_yaw,
        raw_yaw_hp,
        raw_yaw_kl,
        trend,
        info,
    ) = estimate_right_knee(df, RIGHT_KNEE_CFG, dt)

    print("\nRange enforcement diagnostics:")
    print("RAW :", info["raw"])
    print("High-pass:", info["hp"])
    print("KL/SSA   :", info["kl"])
    print("Final method:", info["final_method"])

    # -----------------------------
    # Save final results to CSV
    # knee_angle_final follows USE_HPF:
    #   USE_HPF=True  -> HPF + range shift
    #   USE_HPF=False -> RAW + range shift, no HPF
    # -----------------------------
    out_df = pd.DataFrame({
        "Timestamp": t,
        "knee_angle_final": angles_final,
        "knee_angle_raw_range_shifted": angles_raw_rs,
        "knee_angle_hpf": angles_hp,
        "knee_angle_kl": angles_kl,
    })

    method_tag = "hpf" if USE_HPF else "no_hpf"
    out_path = os.path.join(os.path.dirname(target_file), f"right_knee_angles_{method_tag}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved final knee angles to: {out_path}")
    print(f"Final angle column: knee_angle_final ({info['final_method']})")

    # -----------------------------
    # Plot 1: RAW angles (no drift removal) — separate plot
    # -----------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(t, angles_raw, label="RAW (no drift removal)", linewidth=2)
    plt.title("Right Knee Angle — RAW (no drift removal)")
    plt.xlabel("Time")
    plt.ylabel("Angle (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Plot 2: selected final result
    plt.figure(figsize=(12, 4))
    plt.plot(t, angles_final, label=f"Final: {info['final_method']}", linewidth=2)
    plt.title(
        f"Right Knee Angle — Final selected output (target [{RIGHT_KNEE_CFG['target_min']:.0f}, {RIGHT_KNEE_CFG['target_max']:.0f}] deg)"
    )
    plt.xlabel("Time")
    plt.ylabel("Angle (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Plot 3: Overlay — RAW (range-shifted) vs SSA detrended (range-shifted)
    plt.figure(figsize=(12, 4))
    plt.plot(t, angles_raw_rs, label="RAW + range shift", alpha=0.85)
    plt.plot(t, angles_kl, label="SSA detrended + range shift", linewidth=2)
    plt.title(
        f"Right Knee Angle Overlay (target [{RIGHT_KNEE_CFG['target_min']:.0f}, {RIGHT_KNEE_CFG['target_max']:.0f}] deg)"
    )
    plt.xlabel("Time")
    plt.ylabel("Angle (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
