import os
import time
import math
import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
import xml.etree.ElementTree as ET

class Calibrator:
    def __init__(self):
        # Streams/resolution
        self.W = 640
        self.H = 480

        # Output dirs
        self.vis_dir = "./data/calibration_vis"
        self.calibrate_result_dir = "./assets/calib_params"
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.calibrate_result_dir, exist_ok=True)

        # Enumerate all RealSense devices
        ctx = rs.context()
        self.serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
        self.n_fixed_cameras = len(self.serial_numbers)
        print(f'Found {self.n_fixed_cameras} fixed cameras.')

        # Start one pipeline per device (color+depth)
        self._rs_pipes = {}
        self._rs_profiles = {}
        self._depth_scale = {}
        for serial in self.serial_numbers:
            cfg = rs.config()
            cfg.enable_device(serial)
            
            cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, 30)

            pipe = rs.pipeline()
            profile = pipe.start(cfg)

            self._rs_pipes[serial] = pipe
            self._rs_profiles[serial] = profile
            depth_sensor = profile.get_device().first_depth_sensor()

            self._depth_scale[serial] = float(depth_sensor.get_depth_scale())

        # --- UPDATED API: Initialize CharucoDetector once ---
        self.calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.calibration_board = cv2.aruco.CharucoBoard(
            size=(12, 9),           # Grid size to match 600x400mm
            squareLength=0.03,       # 40mm square length in meters
            markerLength=0.022,       # 30mm marker length (75% of square size)
            dictionary=self.calibration_dictionary,
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.charuco_params = cv2.aruco.CharucoParameters()
        
        # New OpenCV 4.7+ Detector
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.calibration_board, self.charuco_params, self.detector_params
        )

        # Placeholders for extrinsics
        self.R_cam2world = None
        self.t_cam2world = None

    def get_obs(self, get_color=True, get_depth=False) -> dict:
        """
        ONE synchronized frame per camera (depth aligned to color if requested).
        Returns keys: color_{i} [H,W,3 BGR], depth_{i} [H,W float meters], timestamp
        """
        align_to_color = rs.align(rs.stream.color) if get_depth else None
        obs = {}
        timestamp = time.time()

        # Run dummy loop to flush buffer (auto-exposure settle)
        for _ in range(1):
            for cam_idx, serial in enumerate(self.serial_numbers):
                pipe = self._rs_pipes[serial]
                frames = pipe.wait_for_frames()
                if get_depth:
                    frames = align_to_color.process(frames)

                if get_color:
                    color_frame = frames.get_color_frame()
                    color_img = np.asanyarray(color_frame.get_data())
                    obs[f"color_{cam_idx}"] = [color_img]

                if get_depth:
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_scale = self._depth_scale[serial]
                    depth_m = depth_image.astype(np.float32) * depth_scale
                    obs[f"depth_{cam_idx}"] = [depth_m]

        obs["timestamp"] = timestamp
        return obs

    def get_intrinsics(self):
        """
        List of 3x3 K matrices in same order as self.serial_numbers.
        """
        intrinsics = {}
        for serial in self.serial_numbers:
            profile = self._rs_profiles[serial]
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            K = np.array([[intr.fx, 0.0, intr.ppx],
                          [0.0, intr.fy, intr.ppy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            intrinsics[serial] = K
        return intrinsics

    def calibrate(self, visualize=True):
        intrinsics = self.get_intrinsics()
        camera_params = {}
        dist_coef = np.zeros(5, dtype=np.float64)

        for i in range(self.n_fixed_cameras):

            serial_number = self.serial_numbers[i]
            print("Calibrate on serial_num:", serial_number)
            intrinsic = intrinsics[serial_number]

            c2ws = []
            flag = True
            
            while flag:
                flag = False

                obs = self.get_obs(get_color=True, get_depth=True)
                colors = [obs[f"color_{k}"][0] for k in range(self.n_fixed_cameras)]
                
                # Current camera image
                color_bgr = colors[i]
                
                # --- UPDATED API: Single call detection ---
                charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(color_bgr)

                # Check if we found enough corners
                if charuco_corners is None or len(charuco_corners) < 4:
                    print(f"Detection failed charuco_corners: {len(charuco_corners)} or not enough corners. Retrying...")
                    flag = True
                    continue

                # Prepare object points (3D) from the board definition
                obj_points = self.calibration_board.getChessboardCorners()[charuco_ids.flatten(), :]

                # Estimate Pose using solvePnP
                retval, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    charuco_corners,
                    intrinsic,
                    dist_coef,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not retval:
                    print("Pose estimation failed (retval=False). Retrying...")
                    flag = True
                    continue
                else:
                    print("Get Retval success")
                
                # Reproject to calculate error
                reprojected_points, _ = cv2.projectPoints(
                    obj_points,
                    rvec,
                    tvec,
                    intrinsic,
                    dist_coef,
                )

                # Reshape for easier calculation
                reprojected_points = reprojected_points.reshape(-1, 2)
                charuco_corners_flat = charuco_corners.reshape(-1, 2)
                
                # Calculate RMSE
                error = np.sqrt(
                    np.sum((reprojected_points - charuco_corners_flat) ** 2, axis=1)
                ).mean()

                print("Reprojection Error:", error)

                # Threshold check
                if error > 0.3 or len(charuco_corners) < 35:
                    flag = True
                    print("Error too high or points too low. Please try again.")
                    continue
            
            print("Success Project")
            
            # rvec from solvePnP is typically board->cam
            R_board2cam = cv2.Rodrigues(rvec)[0]
            t_board2cam = tvec[:, 0]
            
            # Construct Extrinsic Matrix
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R_board2cam
            extrinsic[:3, 3] = t_board2cam
            
            # (Optional) Store inverse if needed for logic elsewhere
            w2c = extrinsic.copy()
            c2ws.append(np.linalg.inv(w2c))

            # Save into dict
            camera_params[serial_number] = {
                "extrinsic": extrinsic,
                "intrinsic": intrinsics[serial_number],
            }

            if visualize:
                vis_markers = color_bgr.copy()
                # Draw Markers (using the ones returned by detectBoard)
                if marker_corners is not None and len(marker_corners) > 0:
                    cv2.aruco.drawDetectedMarkers(vis_markers, marker_corners, marker_ids)
                
                # Draw Charuco Corners
                if charuco_corners is not None and len(charuco_corners) > 0:
                    cv2.aruco.drawDetectedCornersCharuco(vis_markers, charuco_corners, charuco_ids)
                    
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{serial_number}.jpg', vis_markers)

                axes_img = color_bgr.copy()
                cv2.drawFrameAxes(axes_img, intrinsic, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{serial_number}.jpg", axes_img)

        with open(f'{self.calibrate_result_dir}/cam_params.pkl', "wb") as f:
            print("camera_params:", camera_params)
            pickle.dump(camera_params, f)

def main():
    calib = Calibrator()
    calib.calibrate()
    
    with open("./assets/calib_params/cam_params.pkl", "rb") as f:
        camera_params = pickle.load(f)

if __name__ == "__main__":
    main()