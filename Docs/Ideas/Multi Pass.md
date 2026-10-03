# Offline Two-Pass Calibration + EKF Pipeline for Knee Angle Estimation

Yes — this is actually the **ideal approach** for biomechanics research. Since you're not constrained by real-time processing, you can use the entire dataset to extract high-quality calibration parameters, then run the EKF with those parameters **fixed**. This is standard practice in gait analysis and gives you far better results than trying to estimate everything online.

The workflow has two distinct passes:

---

## 1.  Pass 1: Calibration (Data-Driven Parameter Extraction)

In this pass, you extract all the time-invariant parameters that the EKF needs. You do **not** run the filter yet.

### 1.1.  Static Bias Calibration

If your dataset starts with a **static standing period** (common in gait datasets — subject stands still for 5–10 seconds before running):

- **Gyro bias:** Average the gyro readings during the static period.  
  $$
  \hat{\mathbf{b}}_{g,T} = \frac{1}{N}\sum_{k=1}^{N} \boldsymbol{\omega}_{m,T,k}, \quad \hat{\mathbf{b}}_{g,S} = \frac{1}{N}\sum_{k=1}^{N} \boldsymbol{\omega}_{m,S,k}
  $$
- **Accel bias:** If you have a known reference orientation, you can estimate accel bias. If not, assume it's zero (most MEMS accelerometers are factory-calibrated for bias, and running datasets often pre-calibrate this).
- **Initial orientation:** Use the static period's accelerometer reading to compute initial roll and pitch (gravity direction). If a magnetometer is available, use it for initial yaw; otherwise set yaw to zero.

If there's **no static period**, you can still estimate gyro bias from the **stance phase** of running. During mid-stance (foot flat on ground), the shank angular velocity is near zero for a brief window. Detect these windows using a threshold on gyro magnitude and average across many steps.

### 1.2.  Joint Axis Estimation (The Key Calibration)

This is the most important calibration step. The knee's hinge axis in the **thigh IMU frame** (\( \mathbf{r}_T \)) and in the **shank IMU frame** (\( \mathbf{r}_S \)) can be estimated from the data without any optical reference.

**Method: Relative Angular Velocity Parallelism**

The principle: during pure knee flexion/extension, the relative angular velocity between shank and thigh is aligned with the joint axis. Even during running, the dominant motion is flexion/extension, so the relative angular velocity vectors will cluster around the joint axis.

**Algorithm:**

1. Compute the relative angular velocity in the thigh frame at each time step:
$$
   \mathbf{m}_k = \mathbf{R}_{T,k}^T \mathbf{R}_{S,k} (\boldsymbol{\omega}_{m,S,k} - \hat{\mathbf{b}}_{g,S}) - (\boldsymbol{\omega}_{m,T,k} - \hat{\mathbf{b}}_{g,T})
$$
   Note: for this step, you need orientation estimates. You can get rough orientations by running a **simple complementary filter** or even just integrating the gyro over short windows. The joint axis estimation is robust to moderate orientation errors because it relies on the *direction* of \( \mathbf{m}_k \), not its magnitude.

2. Stack all relative angular velocity vectors into a matrix:
$$
   \mathbf{M} = \begin{bmatrix} \mathbf{m}_1 & \mathbf{m}_2 & \cdots & \mathbf{m}_N \end{bmatrix}^T \in \mathbb{R}^{N \times 3}
$$

3. Perform **Principal Component Analysis (PCA)** or **Singular Value Decomposition (SVD)** on \( \mathbf{M} \). The first principal component (the direction of maximum variance) is the joint axis in the thigh frame:
$$
   \hat{\mathbf{r}}_T = \text{first column of } \mathbf{V} \text{ from } \mathbf{M} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
$$
   The sign is arbitrary; fix it by ensuring \( \hat{\mathbf{r}}_T \) points in a consistent direction (e.g., from medial to lateral, or such that knee flexion is positive).

4. Transform to the shank frame using the mean relative rotation:
$$
   \hat{\mathbf{r}}_S = \bar{\mathbf{R}}_{TS} \hat{\mathbf{r}}_T
$$
   where $\bar{\mathbf{R}}_{TS}$ is the average relative rotation over the dataset.

**Why this works:** During running, the knee undergoes large flexion/extension (the dominant motion), so the relative angular velocity is mostly along the hinge axis. The orthogonal components (from soft-tissue artifact, sensor noise, and small out-of-plane motion) are much smaller and get averaged out by PCA.

**Refinement:** You can iteratively refine the axis. After the first EKF run, re-estimate the axis using the filter's orientation estimates (which are more accurate), then re-run the EKF. This is a **two-iteration** approach that converges quickly.

### 1.3.  Sensor-to-Segment Alignment (Optional but Recommended)

If you have a **functional calibration movement** in the dataset (e.g., a slow knee bend, a squat, or a known pose), you can estimate the rotation from the IMU frame to the anatomical segment frame.

For running specifically, you can use the **static standing period** to align the IMU's vertical axis with the anatomical vertical (gravity). The joint axis estimation (Section 1.2) gives you the medial-lateral axis. The cross product gives you the third axis:

\[
\mathbf{a}_{\text{vertical}} = \text{gravity direction in IMU frame (from accel)}
\]
\[
\mathbf{a}_{\text{ML}} = \hat{\mathbf{r}}_T \quad \text{(joint axis, roughly medial-lateral)}
\]
\[
\mathbf{a}_{\text{AP}} = \mathbf{a}_{\text{vertical}} \times \mathbf{a}_{\text{ML}} \quad \text{(anterior-posterior)}
\]

These three axes form the sensor-to-segment rotation matrix.

### 1.4.  Noise Covariance Estimation

From the static period (or quiet stance phases):

- **Gyro noise \( \sigma_g^2 \):** Variance of the gyro readings during static periods.
- **Accel noise \( \sigma_a^2 \):** Variance of the accelerometer readings during static periods.
- **Gyro bias random walk \( \sigma_{bg}^2 \):** Estimated from the Allan variance of the gyro during static periods, or set to a small value.
- **Constraint noise \( \mathbf{R}_c \):** Set based on the expected soft-tissue artifact. A good starting point is \( \sigma_c = 0.05\text{--}0.2 \) rad/s, tuned by looking at the residual during the calibration movement.

---

## 2.  Pass 2: EKF with Fixed Calibration

Now you run the EKF with all the parameters from Pass 1 **fixed**. You can choose to:

- **Fix the joint axis** \( \hat{\mathbf{r}}_T \) (recommended if you're confident in the calibration).
- **Fix the gyro biases** (recommended if you have a good static period).
- **Still estimate** the attitude errors and possibly small residual bias corrections.

### 2.1.  Simplified State Vector

If you fix the joint axis and biases, the error state reduces to just the two attitude errors:

\[
\delta \mathbf{x} = \begin{bmatrix} \delta \boldsymbol{\theta}_T \\ \delta \boldsymbol{\theta}_S \end{bmatrix} \in \mathbb{R}^6
\]

This is a **huge simplification** — the covariance is 6×6 instead of 15×15, and the Jacobians are much smaller. The filter is faster and more stable.

If you want to allow small bias corrections (recommended), keep the bias errors in the state:

\[
\delta \mathbf{x} = \begin{bmatrix} \delta \boldsymbol{\theta}_T \\ \delta \boldsymbol{\theta}_S \\ \delta \mathbf{b}_{g,T} \\ \delta \mathbf{b}_{g,S} \end{bmatrix} \in \mathbb{R}^{12}
\]

But **fix the joint axis** (remove \( \delta \mathbf{r}_T \) from the state). This is the sweet spot for offline processing: you get bias adaptation without the complexity of estimating the joint axis online.

### 2.2.  Measurement Updates

Run the EKF with:

1. **Accelerometer updates** (gravity reference) at each time step for both IMUs.
2. **Joint axis constraint updates** at each time step using the fixed \( \hat{\mathbf{r}}_T \).

The constraint Jacobian simplifies because \( \delta \mathbf{r}_T = 0 \):

\[
\mathbf{H}_c = \begin{bmatrix} \mathbf{P} [\hat{\mathbf{R}}_{TS} \boldsymbol{\omega}_S']_\times & -\mathbf{P} \hat{\mathbf{R}}_{TS} [\boldsymbol{\omega}_S']_\times & \mathbf{P} & -\mathbf{P} \hat{\mathbf{R}}_{TS} \end{bmatrix}
\]

where \( \mathbf{P} = I - \hat{\mathbf{r}}_T \hat{\mathbf{r}}_T^T \) is the projection orthogonal to the fixed joint axis.

### 2.3.  Optional: Smoothing Pass

Since you're offline, you can run a **Rauch-Tung-Striebel (RTS) smoother** after the forward EKF pass. This runs the filter backward in time and produces smoothed estimates that are more accurate than the forward-only filter. This is a standard technique in offline biomechanics and can reduce errors by 20–40%.

The RTS smoother equations:

\[
\begin{aligned}
\mathbf{C}_k &= \mathbf{P}_{k|k} \Phi_k^T \mathbf{P}_{k+1|k}^{-1} \\
\hat{\mathbf{x}}_{k|N} &= \hat{\mathbf{x}}_{k|k} + \mathbf{C}_k (\hat{\mathbf{x}}_{k+1|N} - \hat{\mathbf{x}}_{k+1|k}) \\
\mathbf{P}_{k|N} &= \mathbf{P}_{k|k} + \mathbf{C}_k (\mathbf{P}_{k+1|N} - \mathbf{P}_{k+1|k}) \mathbf{C}_k^T
\end{aligned}
\]

For the error-state formulation, the smoother operates on the error state and covariance, then the smoothed errors are injected into the nominal state.

---

## 3.  Complete Offline Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  PASS 1: CALIBRATION                                        │
│                                                             │
│  1. Load raw IMU data (thigh + shank)                       │
│  2. Detect static period → estimate gyro biases             │
│  3. Run rough complementary filter → get initial attitudes  │
│  4. Compute relative angular velocities                     │
│  5. PCA/SVD on relative angular velocities → joint axis     │
│  6. Estimate noise covariances from static data             │
│  7. (Optional) Estimate sensor-to-segment alignment         │
│                                                             │
│  Output: b_g,T, b_g,S, r_T, r_S, σ_g², σ_a², R_c           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PASS 2: EKF (with fixed calibration)                       │
│                                                             │
│  1. Initialize nominal states with calibrated biases        │
│  2. Initialize covariance P_0                               │
│  3. For each time step:                                     │
│     a. Propagate nominal state (gyro integration)           │
│     b. Propagate error covariance (Φ P Φ^T + Q)             │
│     c. Accelerometer update (both IMUs)                     │
│     d. Joint axis constraint update                         │
│     e. Inject error, reset error state                      │
│  4. (Optional) RTS smoother backward pass                   │
│                                                             │
│  Output: q_T(t), q_S(t), knee angle θ(t)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 4.  Practical Tips for Offline Processing

1. **Don't over-engineer the filter.** With good calibration from Pass 1, the EKF in Pass 2 can be simple. Fix the joint axis and biases; only estimate attitude errors. This is more robust than trying to estimate everything online.

2. **Validate the joint axis calibration.** After Pass 1, check that the residual \( \mathbf{e} = \mathbf{m} - (\mathbf{m} \cdot \hat{\mathbf{r}}_T)\hat{\mathbf{r}}_T \) is small during knee flexion/extension. If it's not, the axis is wrong — try a different calibration movement or refine with more data.

3. **Use the RTS smoother.** For offline biomechanics, there's no reason not to. It's a few extra lines of code and gives significantly better joint angle estimates, especially at the start and end of the dataset where the forward filter hasn't fully converged.

4. **Iterate once.** Run the EKF, then use the filtered orientations to re-estimate the joint axis (Pass 1, step 5), then re-run the EKF. This two-iteration approach typically converges to the optimal axis.

5. **Check the innovation sequence.** Plot the constraint innovation \( \mathbf{v}_k = -\mathbf{e}_k \) over time. It should be zero-mean and white if the filter is consistent. If it has a bias, your joint axis is slightly off. If it's correlated, your noise covariances are wrong.

6. **Time synchronization.** If the two IMUs are separate devices, ensure their timestamps are aligned. A time offset of even 10 ms can cause significant errors in the relative angular velocity during running. If you have a common trigger or sync signal, use it. Otherwise, estimate the offset by cross-correlating the gyro signals during a known event (e.g., foot strike).

7. **Handle the first few seconds carefully.** The EKF needs a few steps to converge. Discard the first 1–2 seconds of output, or initialize the covariance large and let it settle.

---

## 5.  Summary

| **Step** | **What You Do** | **Output** |
|---|---|---|
| **Pass 1a** | Static bias estimation | \( \hat{\mathbf{b}}_{g,T}, \hat{\mathbf{b}}_{g,S} \) |
| **Pass 1b** | Joint axis estimation via PCA on relative angular velocities | \( \hat{\mathbf{r}}_T, \hat{\mathbf{r}}_S \) |
| **Pass 1c** | Noise covariance estimation | \( \sigma_g^2, \sigma_a^2, \mathbf{R}_c \) |
| **Pass 1d** | (Optional) Sensor-to-segment alignment | \( \mathbf{R}_{\text{IMU}\to\text{segment}} \) |
| **Pass 2a** | Forward EKF with fixed parameters | \( \mathbf{q}_T(t), \mathbf{q}_S(t) \) |
| **Pass 2b** | (Optional) RTS smoother | Smoothed \( \mathbf{q}_T(t), \mathbf{q}_S(t) \) |
| **Pass 2c** | Knee angle extraction | \( \theta(t) = \boldsymbol{\phi} \cdot \hat{\mathbf{r}}_T \) |

This two-pass approach gives you the best of both worlds: the calibration quality of a batch optimization and the robustness of a recursive filter. It's the standard methodology in offline gait analysis, and it will give you knee angle estimates accurate to within a few degrees without any magnetometer.