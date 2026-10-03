Extracting reliable 3D joint angles from asynchronous, multi-view 2D video requires a pipeline that handles markerless pose estimation, signal-based temporal synchronization, and stereo triangulation. This approach avoids manual frame matching and leverages established computer vision algorithms validated against optical marker systems.

**1.2D Pose Extraction:**Run independently on both unaligned videos.

Process both the front and side videos through a scientifically validated markerless pose estimation framework.
- **OpenPose** or **MediaPipe (BlazePose)** are the standard choices in biomechanics literature for tracking anatomical landmarks.
- Extract the $(u, v)$ pixel coordinates for all major joints (hips, knees, ankles, shoulders, elbows) for every frame. Save these as separate time-series datasets. Do not worry about the frame mismatch yet.

**2.Temporal Synchronization:**Aligning the time-series via cross-correlation.

Instead of manually guessing the visual sync point, use the kinematic data itself to find the exact frame offset.

- Select a fast-moving joint with a clear trajectory in both views (e.g., the vertical $y$-coordinate of the ankle during a foot strike, or the wrist during a swing).
- Compute the cross-correlation between the front-view 1D signal and the side-view 1D signal (e.g., using `scipy.signal.correlate` in Python).
- The peak of the cross-correlation function represents the exact frame lag ($\Delta t$) between the two cameras. Trim the arrays so both datasets start and end on the exact same synchronized frame.

**3.3D Trajectory Reconstruction:**Triangulating coordinates using Direct Linear Transformation (DLT).

To convert the synchronized 2D pixel pairs into real-world 3D coordinates $(X, Y, Z)$, you must establish the spatial relationship between the cameras.
- If you recorded a checkerboard calibration pattern, compute the intrinsic and extrinsic camera matrices.
- If uncalibrated, identify at least 6 known static reference points visible in both the front and side views (corners of a room, a marked calibration wand) to compute the Direct Linear Transformation (DLT) matrix.
- Pass the synchronized 2D points and your projection matrices into a triangulation function (like `cv2.triangulatePoints`) to yield raw 3D spatial coordinates for each joint over time.

**4.Joint Angle Computation:**Vector mathematics and anatomical filtering.

Define the body segments as 3D vectors. For example, the thigh vector is defined from the hip to the knee, and the shank vector from the knee to the ankle.
- Use the dot product and cross product of these adjacent segment vectors to calculate the raw anatomical joint angles (flexion/extension, abduction/adduction).
- Because vision-based tracking introduces high-frequency noise, pass the computed angles through a zero-lag 4th-order Butterworth low-pass filter (cutoff frequency typically between 6 Hz and 10 Hz for human motion). Alternatively, if you are fusing this vision data with IMU kinematics, feed these 3D positions directly into an Error-State Kalman Filter (ESKF) as your observation update to smooth the final angle estimation.