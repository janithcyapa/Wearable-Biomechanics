# Real-Time Hybrid Biomechanics Framework for Injury Prediction and Rehabilitation
> Project By : Multidisciplinary AI Research Centre(MARC), University of Peradeniya

## Overview
This project implements an integrated wearable biomechanics system that unifies rigid-body mechanics with adaptive machine learning to monitor musculoskeletal health. By combining high-fidelity kinematic data from IMU sensor arrays with physics-informed neural networks, the system quantifies internal joint stresses and muscle tensions in real-time.


**Key Features:**
*   **Hybrid Inference Engine:** Integrates Newtonian inverse dynamics with deep learning (LSTM) to estimate individual muscle forces and joint reaction torques.
*   **Physics-Informed Modeling:** Grounded in musculoskeletal architecture and functional morphology to ensure biological plausibility in force estimations.
*   **Real-Time Injury Assessment:** Continuous monitoring of biomechanical "Failure Modes," including joint-angle asymmetry (>10°), muscle-force imbalance (>15%), and peak tension thresholds (e.g., >300N for tricep strain).
*   **Low-Latency Feedback:** Optimized for edge deployment with a target inference latency of <200ms, enabling proactive alerts before injury onset.
*   **Validated Against Gold Standards:** The underlying framework is validated against optical motion capture (OptiTrack) and force plate kinetics, achieving high accuracy ($R^2 = 0.94$) in joint kinematics.

**Hardware Support:**
*   Designed for multi-node IMU configurations.

### Repository Structure Recommendation
For a mechanical engineer-led hybrid project, I recommend organizing your folders to separate the "Physics" math from the "ML" training:

1.  `/hardware`: CAD files for wearable enclosures and PCB layouts.
2.  `/physics_engine`: Python/MATLAB scripts for the rigid-body inverse dynamics (The "Skeleton" layer).
3.  `/ml_pipeline`: Jupyter notebooks for training the LSTM model using OpenSim-augmented data.
4.  `/firmware`: C++ code for real-time inference on your microcontroller (e.g., ESP32/Arduino).
5.  `/docs`: Calibration protocols and ISB alignment guidelines.