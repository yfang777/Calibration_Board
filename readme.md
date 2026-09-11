# Robopil Calibration Board Code

This repository contains code for calibrating Extrinsic parameters of RealSense cameras using a ChArUco board. It detects connected RealSense devices, identifying the board's pose relative to the camera, and saves the calibration results.

## Hardware Requirements

*   **Camera:** Intel RealSense Camera (e.g., D400 series like D415, D435, D455).
*   **Calibration Board:** ChArUco Board.
    *   **Source:** [Calib.io Checkerboard](https://calib.io/products/checkerboard?variant=8112181280815)
    *   **Specifications:**
        *   Dictionary: `DICT_5X5_250`
        *   Grid Size: 12 x 9
        *   Square Length: 0.03 m (30 mm)
        *   Marker Length: 0.022 m (22 mm)

## Installation

Ensure you have Python installed. Install the required dependencies using pip:

```bash
pip install numpy opencv-python pyrealsense2
```

## Usage

1.  **Connect the RealSense camera** to your computer via USB 3.0.
2.  **Place the ChArUco board** in the camera's field of view. Ensure the board is flat and well-lit.
3.  **Run the calibration script:**

    ```bash
    python calibrator.py
    ```

## Functionality

The script performs the following steps:
1.  Enumerates all connected RealSense devices.
2.  Starts color and depth streams for each camera.
3.  Captures a frame and attempts to detect the ChArUco board.
4.  Iteratively optimizes the pose estimation until the reprojection error is acceptable (below 0.3 pixels) and enough corners (>= 35) are detected.
5.  Calculates the **Extrinsic Matrix** (transformation from Board Frame to Camera Frame) and retrieves the **Intrinsic Matrix**.

## Outputs

### D405 direct calibration and overlay

The direct wrist workflow undistorts RGB using the stream's RealSense distortion
model before detecting ChArUco corners or displaying the image. The rectified
image retains the original intrinsic matrix; zero distortion in `solvePnP` is
therefore intentional. The overlay then applies `K_mujoco @ inv(K_real)`.
This intrinsic mapping applies at every depth when camera poses match.

Recalibrate after updating: old wrist poses were fitted to distorted images and
are rejected by the direct overlay. Hold the robot and board still, with the
robot at `ROBOT_JOINTS_DEG` in `digital_twin_camera/mujoco_sync_wrist_direct.py`:

```bash
python calibrator_wrist_direct.py
python -m digital_twin_camera.mujoco_sync_wrist_direct
```

The second command regenerates the camera XML from the new pose. Other camera
workflows retain their existing capture behavior. The direct overlay's unwarped
mode now shows an undistorted image, rather than raw sensor pixels.

The script generates the following outputs:

*   **Visualizations:** Saved in `./data/calibration_vis/`
    *   `calibration_detected_marker_{serial_number}.jpg`: Shows detected markers and corners.
    *   `calibration_result_{serial_number}.jpg`: Shows the coordinate frame axes drawn on the board.
*   **Parameters:** Saved in `./assets/calib_params/`
    *   `cam_params.pkl`: A Python pickle file containing a dictionary with camera serial numbers as keys.
        *   Each entry contains:
            *   `"extrinsic"`: 4x4 Transformation Matrix (Board -> Camera).
            *   `"intrinsic"`: 3x3 Camera Intrinsic Matrix.


Hang_Eye: 
* Mathematics: $$T^{base}_{board} = T^{base}_{ee} x T^{ee}_{wrist} x T^{wrist}_{board} $$.
* Eliminate $T^{base}_{board}$ and calculate $T^{ee}_{wrist}$ through averaging 20 times.
* Convention: World_Frame is set to Robot_base.


Regarding get_position: yes, it is the right pose source for hand-eye calibration, and calibrator_hand_eye.py:468 already
  records it for every accepted capture. I recommend keeping that instead of deriving the TCP pose from joint angles and MuJoCo
  FK. The current FK comparison showed discrepancies up to 5.70 mm and 1.883°.
