import os
import time
import math
import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
import yaml
import xml.etree.ElementTree as ET

def set_if_supported(sensor, option, value):
    if sensor.supports(option):
        r = sensor.get_option_range(option)
        value = max(r.min, min(r.max, value))
        sensor.set_option(option, value)

def configure_color_sensor(color_sensor):
    # Turn off auto settings
    set_if_supported(color_sensor, rs.option.enable_auto_exposure, 1)
    set_if_supported(color_sensor, rs.option.enable_auto_white_balance, 1)
    # set_if_supported(color_sensor, rs.option.enable_auto_gain, 1)


def find_color_sensor(device):
    for sensor in device.sensors:
        name = sensor.get_info(rs.camera_info.name)
        if "RGB" in name or "Color" in name:
            return sensor
    raise RuntimeError("No RGB/color sensor found.")


class Calibrator:
    def __init__(self, enable_depth=False):
        # Streams/resolution
        self.W = 640
        self.H = 480
        self.enable_depth = enable_depth
        self.frame_timeout_ms = 5000
        self.frame_wait_attempts = 1
        self.stream_restart_attempts = 3

        # Output dirs
        self.vis_dir = "./data/calibration_vis"
        self.calibrate_result_dir = "./data/calib_params"
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.calibrate_result_dir, exist_ok=True)

        # Enumerate all RealSense devices
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.configured_rs_ids = {
            str(rs_id) for rs_id in config["rs_id"]["d455"].values()
        }

        ctx = rs.context()
        connected_serial_numbers = [
            dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()
        ]
        self.serial_numbers = [
            serial
            for serial in connected_serial_numbers
            if serial in self.configured_rs_ids
        ]
        self.n_fixed_cameras = len(self.serial_numbers)
        camera_suffix = "" if self.n_fixed_cameras == 1 else "s"
        print(f"Found {self.n_fixed_cameras} configured fixed camera{camera_suffix}.")

        # Calibration does not need synchronized multi-camera frames. Start one
        # RealSense pipeline at a time to avoid USB/backend stalls.
        self._rs_pipes = {}
        self._rs_profiles = {}
        self._depth_scale = {}

        self.calibration_dictionary = None
        self.calibration_board = None
        self.detector_params = None
        self.charuco_params = None
        self.charuco_detector = None

        # Placeholders for extrinsics
        self.R_cam2world = None
        self.t_cam2world = None

    def _init_charuco_detector(self):
        self.calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.calibration_board = cv2.aruco.CharucoBoard(
            size=(12, 9),           # Grid size to match 600x400mm
            squareLength=0.03,       # 30mm square length in meters
            markerLength=0.022,      # 22mm marker length
            dictionary=self.calibration_dictionary,
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.calibration_board, self.charuco_params, self.detector_params
        )

    def _clear_charuco_detector(self):
        self.charuco_detector = None
        self.charuco_params = None
        self.detector_params = None
        self.calibration_board = None
        self.calibration_dictionary = None

    def _start_camera(self, serial):
        if serial in self._rs_pipes:
            return

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 30)
        if self.enable_depth:
            cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, 30)

        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        try:
            device = profile.get_device()
            color_sensor = find_color_sensor(device)
            configure_color_sensor(color_sensor)

            if self.enable_depth:
                depth_sensor = device.first_depth_sensor()
                self._depth_scale[serial] = float(depth_sensor.get_depth_scale())
        except Exception:
            pipe.stop()
            raise

        self._rs_pipes[serial] = pipe
        self._rs_profiles[serial] = profile

    def _stop_camera(self, serial):
        pipe = self._rs_pipes.pop(serial, None)
        if pipe is None:
            return
        try:
            pipe.stop()
        finally:
            self._rs_profiles.pop(serial, None)
            self._depth_scale.pop(serial, None)

    def close(self):
        for serial in list(self._rs_pipes):
            self._stop_camera(serial)

    def _restart_camera(self, serial):
        self._stop_camera(serial)
        time.sleep(1.0)
        self._start_camera(serial)

    def _wait_for_frames(self, pipe, serial):
        last_error = None
        for attempt in range(1, self.frame_wait_attempts + 1):
            try:
                return pipe.wait_for_frames(self.frame_timeout_ms)
            except RuntimeError as exc:
                last_error = exc
                print(
                    f"Camera {serial} frame wait timed out "
                    f"({attempt}/{self.frame_wait_attempts})."
                )
        raise RuntimeError(
            f"Camera {serial} did not deliver frames after "
            f"{self.frame_wait_attempts} attempts."
        ) from last_error

    def get_obs(self, get_color=True, get_depth=False) -> dict:
        """
        ONE synchronized frame per camera (depth aligned to color if requested).
        Returns keys: color_{i} [H,W,3 BGR], depth_{i} [H,W float meters], timestamp
        """
        if get_depth and not self.enable_depth:
            raise ValueError("Depth was requested, but this Calibrator was started without depth.")

        align_to_color = rs.align(rs.stream.color) if get_depth else None
        obs = {}
        timestamp = time.time()

        # Run dummy loop to flush buffer (auto-exposure settle)
        for _ in range(1):
            for serial, pipe in list(self._rs_pipes.items()):
                cam_idx = self.serial_numbers.index(serial)
                for restart_attempt in range(self.stream_restart_attempts + 1):
                    try:
                        frames = self._wait_for_frames(pipe, serial)
                        break
                    except RuntimeError:
                        if restart_attempt >= self.stream_restart_attempts:
                            raise
                        print(f"Restarting camera {serial} after frame timeout.")
                        self._restart_camera(serial)
                        pipe = self._rs_pipes[serial]

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
        3x3 K matrices for active camera pipelines.
        """
        intrinsics = {}
        for serial in self._rs_profiles:
            intrinsics[serial] = self.get_intrinsic(serial)
        return intrinsics

    def get_intrinsic(self, serial):
        profile = self._rs_profiles[serial]
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        return np.array([[intr.fx, 0.0, intr.ppx],
                         [0.0, intr.fy, intr.ppy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def calibrate(self, visualize=True):
        camera_params = {}
        dist_coef = np.zeros(5, dtype=np.float64)

        for i in range(self.n_fixed_cameras):

            serial_number = self.serial_numbers[i]
            if serial_number not in self.configured_rs_ids:
                print("Skip serial_num not in config:", serial_number)
                continue
            print("Calibrate on serial_num:", serial_number)
            self._clear_charuco_detector()
            self._start_camera(serial_number)
            intrinsic = self.get_intrinsic(serial_number)

            c2ws = []
            flag = True

            try:
                while flag:
                    flag = False

                    # Current camera image
                    obs = self.get_obs(get_color=True, get_depth=False)
                    color_bgr = obs[f"color_{i}"][0]

                    try:
                        self._init_charuco_detector()

                        # --- UPDATED API: Single call detection ---
                        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(color_bgr)

                        # Check if we found enough corners
                        num_corners = 0 if charuco_corners is None else len(charuco_corners)
                        if num_corners < 4:
                            print(f"Detection failed charuco_corners: {num_corners} or not enough corners. Retrying...")
                            flag = True
                            continue

                        # Prepare object points (3D) from the board definition
                        obj_points = self.calibration_board.getChessboardCorners()[charuco_ids.flatten(), :]
                    finally:
                        self._clear_charuco_detector()

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
                    print("charuco_corners:", len(charuco_corners))

                    # Threshold check
                    if error > 0.3 or len(charuco_corners) < 35:
                        flag = True
                        print("Error too high or points too low. Please try again.")
                        continue
            finally:
                self._clear_charuco_detector()
                self._stop_camera(serial_number)
            
            print("Success Project")
            
            # rvec from solvePnP is typically board->cam
            R_board2cam = cv2.Rodrigues(rvec)[0]
            t_board2cam = tvec[:, 0]

            # Redefine world frame as Rx(180°) relative to the board's native frame
            # (flip Y-Z). Right-multiply rotation; translation is unchanged (shared origin).
            Rx180 = np.diag([1.0, -1.0, -1.0])
            R_board2cam = R_board2cam @ Rx180

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
                "intrinsic": intrinsic,
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

                # Original board frame (pre-flip): uses raw rvec/tvec from solvePnP.
                axes_original = color_bgr.copy()
                cv2.drawFrameAxes(axes_original, intrinsic, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/world_frame_original_{serial_number}.png", axes_original)

                # Converted world frame (post-flip): derive rvec from flipped R_board2cam.
                rvec_converted, _ = cv2.Rodrigues(R_board2cam)
                axes_converted = color_bgr.copy()
                cv2.drawFrameAxes(axes_converted, intrinsic, dist_coef, rvec_converted, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/world_frame_converted_{serial_number}.png", axes_converted)

        with open(f'{self.calibrate_result_dir}/cam_params.pkl', "wb") as f:
            print("camera_params:", camera_params)
            pickle.dump(camera_params, f)

def main():
    calib = Calibrator()
    try:
        calib.calibrate()
    finally:
        calib.close()
    
    with open("./data/calib_params/cam_params.pkl", "rb") as f:
        camera_params = pickle.load(f)

if __name__ == "__main__":
    main()
