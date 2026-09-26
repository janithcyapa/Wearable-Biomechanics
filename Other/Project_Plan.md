## Phase 1: Hardware Integration & Sensor Alignment
**Objective:** Stabilize the 14-IMU body-area network and ensure anatomical accuracy.
*   **Sensor Calibration:** Perform static T-pose alignment and zero-offset correction for all 14 nodes to achieve angular accuracy below 1.2°[cite: 1].
*   **Anatomical Mapping:** Align sensors with International Society of Biomechanics (ISB) guidelines on the trunk, upper arms, forearms, thighs, shanks, and feet[cite: 1, 2].
*   **Drift Compensation:** Implement a Bluetooth timestamp-alignment protocol with drift compensation every 5 seconds to prevent skeleton "twisting"[cite: 1].

## Phase 2: Data Acquisition & Segmentation
**Objective:** Collect raw lifting data and break it into usable segments.
*   **Pilot Lifting Sessions:** Record continuous sessions of various lifts (Squat, Deadlift, Clean) while documenting the physical barbell weight[cite: 2].
*   **Repetition Segmentation:** Use a threshold-based algorithm on vertical acceleration to isolate individual repetitions from the continuous stream[cite: 1, 2].
*   **Phase Identification:** Implement velocity zero-crossing detection to segment each rep into **Eccentric** (lowering) and **Concentric** (lifting) phases[cite: 1].
*   **FFT Feature Extraction:** Transform segmented cycles into the frequency domain (Fast Fourier Transform), retaining ~11 coefficients per axis to compress the motion data[cite: 2].

## Phase 3: Biomechanical Modeling (Stage 1 Hybrid)
**Objective:** Build the "Physics" layer of your model using Inverse Dynamics.
*   **Anthropometric Scaling:** Estimate segment masses and moments of inertia using standard regression tables (e.g., Winter’s tables) based on total body mass and height[cite: 2].
*   **Rigid-Body Inverse Dynamics:** Use Newton-Euler equations to calculate **Joint Torques** ($Torque = I \cdot \alpha$) for each joint in the 14-node chain[cite: 2].
*   **OpenSim Ground Truth:** Feed your IMU orientations and barbell weights into **OpenSim** (using the TLEM 2.0 model) to generate "actual" internal tissue loads for training[cite: 2].



## Phase 4: Machine Learning & Scoring Development
**Objective:** Train the models that estimate force and predict risk.
*   **Force Regression (The "Hybrid" Model):** Train a model (Elastic Net or Random Forest) to map your calculated Joint Torques to individual **Muscle Tensions** and **Bone Compressions**[cite: 2].
*   **Technique Classification:** Use a **Bi-directional LSTM** (Long Short-Term Memory) network to classify movement types and identify "sticking points" in the lift[cite: 1].
*   **Injury/Risk Scoring:** 
    *   **Deterministic:** Flag risks when predicted tension exceeds biological limits (e.g., >250N for lumbar stress)[cite: 1].
    *   **Probabilistic:** Use the LSTM to predict high-risk conditions (like valgus collapse >10°) 1.4 to 2.6 seconds before they occur[cite: 1].

## Phase 5: Validation & Benchmarking
**Objective:** Prove your system works without a 9-camera setup.
*   **Computer Vision Benchmark:** Record trials with 2-3 phone cameras and process the footage through **MediaPipe** or **OpenCap** to validate your IMU joint angles[cite: 1, 2].
*   **Statistical Concordance:** Perform **Bland-Altman Analysis** and calculate the **Intraclass Correlation Coefficient (ICC)** to ensure your IMU system agrees with the visual/OpenSim ground truth[cite: 1].
*   **Asymmetry Evaluation:** Calculate a Left-Right Symmetry Index to detect imbalances >15% during heavy lifts[cite: 1].

## Phase 6: Reporting & Deployment
**Objective:** Finalize the GitHub Repo and MARC UOP documentation.
*   **OpenSim Final Report:** Export simulated load profiles for the "best players" vs. amateur lifters to serve as your project reference[cite: 2].
*   **GitHub Documentation:** Finalize the repo with folders for `/physics_engine`, `/ml_pipeline`, and `/firmware`.
*   **Edge Optimization:** Convert the trained LSTM/Regression models to C++ header files for real-time inference on your wearable hardware[cite: 1].

