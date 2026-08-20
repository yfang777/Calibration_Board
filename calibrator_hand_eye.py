"""Capture and solve an xArm7 wrist-camera hand-eye calibration.

The calibration target must remain fixed in the robot base/world frame and be
visible to both the configured fixed D455 and wrist-mounted D405. ``capture``
moves the physical robot through 20 joint poses, saving a paired image and the
measured xArm TCP pose at every stop. ``solve`` is hardware-free and estimates
the wrist-camera-to-end-effector transform from the saved dataset.

Examples:

    python calibrator_hand_eye.py --mode capture
    python calibrator_hand_eye.py --mode solve
    python calibrator_hand_eye.py --mode all

Capture requires an explicit MOVE confirmation unless ``--yes`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml


REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO / "config.yaml"
DEFAULT_OUTPUT_DIR = REPO / "data/hand_eye"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_FPS = 30

# These values intentionally match calibrator.py.
BOARD_DICTIONARY_NAME = "DICT_5X5_250"
BOARD_SIZE = (12, 9)
BOARD_SQUARE_M = 0.03
BOARD_MARKER_M = 0.022
DISTORTION = np.zeros(5, dtype=np.float64)
WORLD_TO_NATIVE_BOARD = np.diag([1.0, -1.0, -1.0])

BOARD_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
CALIBRATION_BOARD = cv2.aruco.CharucoBoard(
    size=BOARD_SIZE,
    squareLength=BOARD_SQUARE_M,
    markerLength=BOARD_MARKER_M,
    dictionary=BOARD_DICTIONARY,
)
CHARUCO_DETECTOR = cv2.aruco.CharucoDetector(
    CALIBRATION_BOARD,
    cv2.aruco.CharucoParameters(),
    cv2.aruco.DetectorParameters(),
)

INITIAL_JOINTS_DEG = np.array([0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0])

# First 20 tested wrist-sweep poses from ref.py. Keep the count fixed: capture
# accepts a dataset only after all 20 poses have a valid paired observation.
CAPTURE_JOINTS_DEG = np.array(
    [
        [0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0],
        [0.0, -25.5, 0.1, 42.6, -0.1, 68.0, 0.0],
        [3.2, -20.6, -2.7, 46.4, -1.1, 66.6, -20.4],
        [-4.0, -20.4, 3.7, 46.8, 1.3, 67.7, 26.6],
        [-17.7, -29.9, -12.0, 40.4, -6.5, 69.7, -8.7],
        [-7.4, -20.6, -17.8, 47.9, -6.5, 66.9, -43.7],
        [15.4, -23.3, 14.2, 45.6, 5.8, 68.5, 4.4],
        [4.9, -4.0, 19.1, 63.7, 1.9, 68.5, 59.8],
        [0.0, -12.3, 0.2, 54.0, -0.2, 66.3, 0.0],
        [0.1, 14.1, 0.3, 88.3, 0.5, 95.7, -0.1],
        [0.1, 19.1, 0.4, 96.1, 0.8, 109.9, 0.0],
        [-0.3, -50.9, 0.1, 23.0, -0.3, 57.5, 0.0],
        [-0.4, -81.1, -0.1, 7.9, -0.4, 53.8, 0.0],
        [0.1, -28.3, 0.1, 56.9, 0.2, 85.3, 0.0],
        [-0.3, -17.3, 0.4, 26.5, -0.8, 44.0, 0.2],
        [7.8, -21.6, 14.5, 63.0, 5.6, 84.6, 20.5],
        [5.5, -18.1, 21.5, 67.8, -14.4, 95.6, 25.2],
        [4.7, -3.4, 21.0, 81.2, -19.7, 95.3, 25.3],
        [9.1, 8.3, 17.5, 96.9, -16.1, 116.5, 18.7],
        [13.6, 11.1, 10.7, 69.0, -15.6, 86.3, 25.0],
    ],
    dtype=np.float64,
)

HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class BoardObservation:
    camera_from_board: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    marker_corners: Sequence[np.ndarray]
    marker_ids: np.ndarray | None
    reprojection_rmse_px: float

    @property
    def corner_count(self) -> int:
        return len(self.charuco_ids)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required calibration file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"Rotation must have shape (3, 3), received {rotation.shape}.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _validated_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"{name} must have shape (4, 4), got {transform.shape}.")
    if not np.all(np.isfinite(transform)):
        raise RuntimeError(f"{name} contains non-finite values.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError(f"{name} has an invalid homogeneous bottom row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise RuntimeError(f"{name} rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise RuntimeError(f"{name} rotation is not a proper rotation.")
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must have shape (4, 4), received {transform.shape}.")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert xArm roll/pitch/yaw radians to Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def _xarm_pose_to_transform(tcp_pose: Sequence[float]) -> np.ndarray:
    """Convert xArm [x,y,z mm, roll,pitch,yaw rad] to base_from_ee."""
    pose = np.asarray(tcp_pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected a six-element xArm TCP pose, received {pose.shape}.")
    rotation = _rpy_to_rotation(*pose[3:])
    return _make_transform(rotation, pose[:3] / 1000.0)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _average_rotations(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("At least one rotation is required.")
    mean_rotation = np.mean(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("At least one transform is required.")
    rotation = _average_rotations([transform[:3, :3] for transform in transforms])
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    return _make_transform(rotation, translation)


def _transform_errors(
    reference: np.ndarray,
    estimates: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    translation_errors = []
    rotation_errors = []
    for estimate in estimates:
        delta = _invert_transform(reference) @ estimate
        translation_errors.append(np.linalg.norm(delta[:3, 3]))
        rotation_errors.append(_rotation_angle_deg(delta[:3, :3]))
    return (
        np.asarray(translation_errors, dtype=np.float64),
        np.asarray(rotation_errors, dtype=np.float64),
    )


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def _motion_spans(transforms: Sequence[np.ndarray]) -> tuple[float, float]:
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for first_index, first in enumerate(transforms):
        for second in transforms[first_index + 1 :]:
            relative = _invert_transform(first) @ second
            maximum_translation = max(
                maximum_translation,
                float(np.linalg.norm(relative[:3, 3])),
            )
            maximum_rotation = max(
                maximum_rotation,
                _rotation_angle_deg(relative[:3, :3]),
            )
    return maximum_translation, maximum_rotation


def estimate_board_pose(
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
) -> BoardObservation:
    """Estimate the native ChArUco board-to-camera transform."""
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("A BGR camera image is required for ChArUco detection.")
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must have shape (3, 3), got {intrinsic.shape}.")

    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        CHARUCO_DETECTOR.detectBoard(image_bgr)
    )
    corner_count = 0 if charuco_corners is None else len(charuco_corners)
    if charuco_ids is None or corner_count < 4:
        raise RuntimeError(
            f"ChArUco detection found {corner_count} corners; at least four are required."
        )

    ids = charuco_ids.reshape(-1).astype(np.int64)
    board_points = np.asarray(
        CALIBRATION_BOARD.getChessboardCorners(), dtype=np.float32
    )
    if np.any(ids < 0) or np.any(ids >= len(board_points)):
        raise RuntimeError("ChArUco detection returned an out-of-range corner ID.")
    object_points = np.ascontiguousarray(board_points[ids])
    image_points = np.ascontiguousarray(charuco_corners, dtype=np.float32)
    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsic,
        DISTORTION,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise RuntimeError("OpenCV solvePnP could not estimate the ChArUco pose.")

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsic,
        DISTORTION,
    )
    residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    reprojection_rmse_px = float(
        np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))
    )
    rotation, _ = cv2.Rodrigues(rvec)
    return BoardObservation(
        camera_from_board=_make_transform(rotation, tvec),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32),
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32),
        marker_corners=marker_corners,
        marker_ids=marker_ids,
        reprojection_rmse_px=reprojection_rmse_px,
    )


def _observation_is_acceptable(
    observation: BoardObservation,
    *,
    min_corners: int,
    max_reprojection_error_px: float,
) -> bool:
    return (
        observation.corner_count >= min_corners
        and observation.reprojection_rmse_px <= max_reprojection_error_px
    )


def _annotate_observation(
    image_bgr: np.ndarray,
    observation: BoardObservation | None,
    intrinsic: np.ndarray,
    label: str,
    error: str | None = None,
) -> np.ndarray:
    annotated = image_bgr.copy()
    if observation is not None:
        if observation.marker_corners is not None and len(observation.marker_corners):
            cv2.aruco.drawDetectedMarkers(
                annotated,
                observation.marker_corners,
                observation.marker_ids,
            )
        cv2.aruco.drawDetectedCornersCharuco(
            annotated,
            observation.charuco_corners,
            observation.charuco_ids,
        )
        cv2.drawFrameAxes(
            annotated,
            intrinsic,
            DISTORTION,
            observation.rvec,
            observation.tvec,
            BOARD_SQUARE_M * 3.0,
        )
        status = (
            f"{label}: {observation.corner_count} corners, "
            f"{observation.reprojection_rmse_px:.3f}px"
        )
        color = (0, 255, 0)
    else:
        status = f"{label}: {error or 'detection failed'}"
        color = (0, 0, 255)
    cv2.putText(
        annotated,
        status,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        status,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV could not write image: {path}")


def _load_camera_serials(
    config_path: Path,
    fixed_override: str | None,
    wrist_override: str | None,
) -> tuple[str, str]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        fixed_values = list(config["rs_id"]["d455"].values())
        configured_wrist = config["rs_id"]["d405"]["cam_1"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Expected rs_id.d455 and rs_id.d405.cam_1 in {config_path}."
        ) from exc
    if not fixed_values and fixed_override is None:
        raise RuntimeError(f"No fixed D455 camera is configured in {config_path}.")
    fixed_serial = str(fixed_override or fixed_values[0])
    wrist_serial = str(wrist_override or configured_wrist)
    if fixed_serial == wrist_serial:
        raise ValueError("The fixed-camera and wrist-camera serials must differ.")
    return fixed_serial, wrist_serial


def _assert_realsense_connected(serials: Sequence[str]) -> None:
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Capture requires pyrealsense2 in the active Python environment."
        ) from exc
    context = rs.context()
    connected = {
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    }
    missing = [serial for serial in serials if serial not in connected]
    if missing:
        raise RuntimeError(
            f"RealSense cameras are not connected: {missing}. "
            f"Connected serials: {sorted(connected)}"
        )


def _read_xarm_state(arm: Any) -> tuple[np.ndarray, np.ndarray, list[float]]:
    pose_code, tcp_pose = arm.get_position(is_radian=True)
    if pose_code != 0:
        raise RuntimeError(f"xArm get_position failed with code {pose_code}.")
    angle_code, measured_joints_deg = arm.get_servo_angle(is_radian=False)
    if angle_code != 0:
        raise RuntimeError(f"xArm get_servo_angle failed with code {angle_code}.")
    base_from_end_effector = _xarm_pose_to_transform(tcp_pose)
    return (
        base_from_end_effector,
        np.asarray(tcp_pose, dtype=np.float64),
        [float(value) for value in measured_joints_deg],
    )


def _move_xarm(
    arm: Any,
    joints_deg: np.ndarray,
    *,
    joint_speed_deg_s: float,
    settle_seconds: float,
) -> None:
    code = arm.set_servo_angle(
        angle=np.asarray(joints_deg, dtype=np.float64).tolist(),
        speed=joint_speed_deg_s,
        is_radian=False,
        wait=True,
    )
    if code != 0:
        raise RuntimeError(f"xArm set_servo_angle failed with code {code}.")
    if settle_seconds > 0.0:
        time.sleep(settle_seconds)


def _capture_confirmation(
    *,
    robot_ip: str,
    fixed_serial: str,
    wrist_serial: str,
    assume_yes: bool,
) -> None:
    print("Capture WILL MOVE the physical xArm through these 20 joint poses:")
    for index, pose in enumerate(CAPTURE_JOINTS_DEG):
        values = " ".join(f"{value:6.1f}" for value in pose)
        print(f"  {index:02d}: [{values}] deg")
    print(f"Robot: {robot_ip}")
    print(f"Fixed camera: {fixed_serial}")
    print(f"Wrist camera: {wrist_serial}")
    print("Verify the workspace is clear and the board is visible from every pose.")
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Capture needs an interactive MOVE confirmation; pass --yes only after "
            "verifying the robot poses and workspace."
        )
    confirmation = input("Type MOVE to start physical robot motion: ").strip()
    if confirmation != "MOVE":
        raise RuntimeError("Capture cancelled; confirmation did not match MOVE.")


def capture_dataset(args: argparse.Namespace) -> Path:
    """Move the robot and capture 20 paired camera observations."""
    fixed_serial, wrist_serial = _load_camera_serials(
        args.config,
        args.fixed_serial,
        args.wrist_serial,
    )
    capture_dir = args.output_dir / "captures"
    visualization_dir = args.output_dir / "visualizations/capture"
    manifest_path = capture_dir / "dataset.json"
    if manifest_path.exists() and not args.overwrite_captures:
        raise FileExistsError(
            f"Capture manifest already exists: {manifest_path}. "
            "Use --overwrite-captures to replace this dataset."
        )
    _capture_confirmation(
        robot_ip=args.robot_ip,
        fixed_serial=fixed_serial,
        wrist_serial=wrist_serial,
        assume_yes=args.yes,
    )
    _assert_realsense_connected([fixed_serial, wrist_serial])

    capture_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    try:
        from digital_twin_camera.robot_env import Xarm7Env
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Capture requires the xArm Python SDK and pyrealsense2 dependencies."
        ) from exc

    environment = None
    capture_error: BaseException | None = None
    try:
        environment = Xarm7Env(
            robot_ip=args.robot_ip,
            gripper_enable=False,
            connect_robot=True,
        )
        environment.set_cam(
            [fixed_serial, wrist_serial],
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            fps=IMAGE_FPS,
            enable_depth=False,
        )
        print("Reading active RealSense color intrinsics...")
        fixed_intrinsic = environment.camera_intrinsics[fixed_serial].copy()
        wrist_intrinsic = environment.camera_intrinsics[wrist_serial].copy()
        if environment.arm is None:
            raise RuntimeError("xArm connection was not initialized.")
        preflight_frames = environment.get_obs()
        if fixed_serial not in preflight_frames or wrist_serial not in preflight_frames:
            raise RuntimeError("Paired RealSense preflight did not return both cameras.")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "board": {
                "dictionary": BOARD_DICTIONARY_NAME,
                "size": list(BOARD_SIZE),
                "square_length_m": BOARD_SQUARE_M,
                "marker_length_m": BOARD_MARKER_M,
            },
            "image": {
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "fps": IMAGE_FPS,
                "distortion": DISTORTION.tolist(),
            },
            "fixed_serial": fixed_serial,
            "wrist_serial": wrist_serial,
            "fixed_intrinsic": fixed_intrinsic,
            "wrist_intrinsic": wrist_intrinsic,
            "robot_ip": args.robot_ip,
            "robot_pose_frame": "base_from_end_effector_tcp",
            "pose_count_expected": len(CAPTURE_JOINTS_DEG),
            "entries": [],
        }
        _write_json_atomic(manifest_path, manifest)

        for pose_index, commanded_joints_deg in enumerate(CAPTURE_JOINTS_DEG):
            print(
                f"Moving to pose {pose_index + 1}/{len(CAPTURE_JOINTS_DEG)}: "
                f"{commanded_joints_deg.tolist()}"
            )
            _move_xarm(
                environment.arm,
                commanded_joints_deg,
                joint_speed_deg_s=args.joint_speed_deg_s,
                settle_seconds=args.settle_seconds,
            )

            accepted = False
            last_fixed_annotation = None
            last_wrist_annotation = None
            last_reason = "no frames were captured"
            for attempt in range(1, args.capture_attempts + 1):
                frames = environment.get_obs()
                fixed_image = frames[fixed_serial]["color"]
                wrist_image = frames[wrist_serial]["color"]

                fixed_observation = None
                wrist_observation = None
                fixed_error = None
                wrist_error = None
                try:
                    fixed_observation = estimate_board_pose(
                        fixed_image, fixed_intrinsic
                    )
                except (RuntimeError, ValueError) as exc:
                    fixed_error = str(exc)
                try:
                    wrist_observation = estimate_board_pose(
                        wrist_image, wrist_intrinsic
                    )
                except (RuntimeError, ValueError) as exc:
                    wrist_error = str(exc)

                fixed_ok = (
                    fixed_observation is not None
                    and _observation_is_acceptable(
                        fixed_observation,
                        min_corners=args.min_corners,
                        max_reprojection_error_px=args.max_reprojection_error_px,
                    )
                )
                wrist_ok = (
                    wrist_observation is not None
                    and _observation_is_acceptable(
                        wrist_observation,
                        min_corners=args.min_corners,
                        max_reprojection_error_px=args.max_reprojection_error_px,
                    )
                )
                last_fixed_annotation = _annotate_observation(
                    fixed_image,
                    fixed_observation,
                    fixed_intrinsic,
                    "fixed",
                    fixed_error,
                )
                last_wrist_annotation = _annotate_observation(
                    wrist_image,
                    wrist_observation,
                    wrist_intrinsic,
                    "wrist",
                    wrist_error,
                )
                if args.show_gui:
                    cv2.imshow(
                        "Hand-eye capture: fixed | wrist",
                        np.hstack((last_fixed_annotation, last_wrist_annotation)),
                    )
                    key = cv2.waitKey(250) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt("Capture cancelled with Escape.")

                if not fixed_ok or not wrist_ok:
                    fixed_summary = (
                        "failed"
                        if fixed_observation is None
                        else (
                            f"{fixed_observation.corner_count} corners, "
                            f"{fixed_observation.reprojection_rmse_px:.3f}px"
                        )
                    )
                    wrist_summary = (
                        "failed"
                        if wrist_observation is None
                        else (
                            f"{wrist_observation.corner_count} corners, "
                            f"{wrist_observation.reprojection_rmse_px:.3f}px"
                        )
                    )
                    last_reason = f"fixed={fixed_summary}; wrist={wrist_summary}"
                    print(
                        f"  Rejected attempt {attempt}/{args.capture_attempts}: "
                        f"{last_reason}"
                    )
                    continue

                base_from_end_effector, measured_tcp_pose, measured_joints_deg = (
                    _read_xarm_state(environment.arm)
                )
                fixed_filename = f"pose_{pose_index:02d}_fixed.png"
                wrist_filename = f"pose_{pose_index:02d}_wrist.png"
                _save_image(capture_dir / fixed_filename, fixed_image)
                _save_image(capture_dir / wrist_filename, wrist_image)
                _save_image(
                    visualization_dir / f"pose_{pose_index:02d}_fixed.png",
                    last_fixed_annotation,
                )
                _save_image(
                    visualization_dir / f"pose_{pose_index:02d}_wrist.png",
                    last_wrist_annotation,
                )
                manifest["entries"].append(
                    {
                        "pose_index": pose_index,
                        "fixed_image": fixed_filename,
                        "wrist_image": wrist_filename,
                        "commanded_joints_deg": commanded_joints_deg,
                        "measured_joints_deg": measured_joints_deg,
                        "measured_tcp_pose_mm_rad": measured_tcp_pose,
                        "base_from_end_effector": base_from_end_effector,
                        "fixed_detection": {
                            "corner_count": fixed_observation.corner_count,
                            "reprojection_rmse_px": (
                                fixed_observation.reprojection_rmse_px
                            ),
                        },
                        "wrist_detection": {
                            "corner_count": wrist_observation.corner_count,
                            "reprojection_rmse_px": (
                                wrist_observation.reprojection_rmse_px
                            ),
                        },
                    }
                )
                _write_json_atomic(manifest_path, manifest)
                print(
                    f"  Accepted: fixed {fixed_observation.corner_count} corners / "
                    f"{fixed_observation.reprojection_rmse_px:.3f}px; wrist "
                    f"{wrist_observation.corner_count} corners / "
                    f"{wrist_observation.reprojection_rmse_px:.3f}px"
                )
                accepted = True
                break

            if not accepted:
                if last_fixed_annotation is not None:
                    _save_image(
                        visualization_dir / f"pose_{pose_index:02d}_rejected_fixed.png",
                        last_fixed_annotation,
                    )
                if last_wrist_annotation is not None:
                    _save_image(
                        visualization_dir / f"pose_{pose_index:02d}_rejected_wrist.png",
                        last_wrist_annotation,
                    )
                raise RuntimeError(
                    f"Pose {pose_index} did not produce an acceptable paired "
                    f"observation after {args.capture_attempts} attempts: {last_reason}"
                )

        manifest["completed_utc"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        print(f"Captured all {len(CAPTURE_JOINTS_DEG)} poses in {capture_dir}.")
        return manifest_path
    except BaseException as exc:
        capture_error = exc
        raise
    finally:
        cv2.destroyAllWindows()
        if environment is not None:
            if environment.arm is not None:
                try:
                    print("Returning the robot to its initial pose...")
                    _move_xarm(
                        environment.arm,
                        INITIAL_JOINTS_DEG,
                        joint_speed_deg_s=args.joint_speed_deg_s,
                        settle_seconds=0.0,
                    )
                except Exception as return_error:
                    print(
                        f"WARNING: could not return the robot to its initial pose: "
                        f"{return_error}",
                        file=sys.stderr,
                    )
                    if capture_error is None:
                        raise
                finally:
                    environment.close()
            else:
                environment.close()


def _solve_hand_eye(
    base_from_end_effector: Sequence[np.ndarray],
    wrist_camera_from_board: Sequence[np.ndarray],
    method_name: str,
) -> np.ndarray:
    method = HAND_EYE_METHODS[method_name]
    rotations_end_effector_to_base = [
        transform[:3, :3] for transform in base_from_end_effector
    ]
    translations_end_effector_to_base = [
        transform[:3, 3].reshape(3, 1) for transform in base_from_end_effector
    ]
    rotations_board_to_camera = [
        transform[:3, :3] for transform in wrist_camera_from_board
    ]
    translations_board_to_camera = [
        transform[:3, 3].reshape(3, 1) for transform in wrist_camera_from_board
    ]
    rotation_camera_to_end_effector, translation_camera_to_end_effector = (
        cv2.calibrateHandEye(
            rotations_end_effector_to_base,
            translations_end_effector_to_base,
            rotations_board_to_camera,
            translations_board_to_camera,
            method=method,
        )
    )
    result = _make_transform(
        rotation_camera_to_end_effector,
        translation_camera_to_end_effector,
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("OpenCV hand-eye calibration returned non-finite values.")
    return result


def _quality_check(name: str, value: float, maximum: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "maximum": maximum,
        "passed": value <= maximum,
    }


def solve_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Solve hand-eye calibration from a previously captured dataset."""
    capture_dir = args.output_dir / "captures"
    visualization_dir = args.output_dir / "visualizations/solve"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = capture_dir / "dataset.json"
    manifest = _load_json(manifest_path)
    board_metadata = manifest.get("board", {})
    expected_board_metadata = {
        "dictionary": BOARD_DICTIONARY_NAME,
        "size": list(BOARD_SIZE),
        "square_length_m": BOARD_SQUARE_M,
        "marker_length_m": BOARD_MARKER_M,
    }
    if board_metadata and board_metadata != expected_board_metadata:
        raise RuntimeError(
            "Dataset ChArUco parameters do not match calibrator.py: "
            f"{board_metadata} versus {expected_board_metadata}."
        )
    entries = manifest.get("entries", [])
    expected_count = int(manifest.get("pose_count_expected", len(CAPTURE_JOINTS_DEG)))
    if expected_count != len(CAPTURE_JOINTS_DEG):
        raise RuntimeError(
            f"Dataset expects {expected_count} poses, but this calibrator requires "
            f"{len(CAPTURE_JOINTS_DEG)}."
        )
    if len(entries) != expected_count:
        raise RuntimeError(
            f"Dataset contains {len(entries)}/{expected_count} completed poses. "
            "Run capture again to obtain all 20 valid observations."
        )
    pose_indices = sorted(int(entry["pose_index"]) for entry in entries)
    if pose_indices != list(range(expected_count)):
        raise RuntimeError(
            f"Dataset pose indices must be 0..{expected_count - 1}, got {pose_indices}."
        )

    fixed_intrinsic = np.asarray(manifest["fixed_intrinsic"], dtype=np.float64)
    wrist_intrinsic = np.asarray(manifest["wrist_intrinsic"], dtype=np.float64)
    base_from_end_effector: list[np.ndarray] = []
    fixed_camera_from_board: list[np.ndarray] = []
    wrist_camera_from_board: list[np.ndarray] = []
    fixed_reprojection_errors = []
    wrist_reprojection_errors = []

    for entry in sorted(entries, key=lambda item: int(item["pose_index"])):
        pose_index = int(entry["pose_index"])
        fixed_image_path = capture_dir / entry["fixed_image"]
        wrist_image_path = capture_dir / entry["wrist_image"]
        fixed_image = cv2.imread(str(fixed_image_path), cv2.IMREAD_COLOR)
        wrist_image = cv2.imread(str(wrist_image_path), cv2.IMREAD_COLOR)
        if fixed_image is None:
            raise RuntimeError(f"Could not read captured image: {fixed_image_path}")
        if wrist_image is None:
            raise RuntimeError(f"Could not read captured image: {wrist_image_path}")
        fixed_observation = estimate_board_pose(fixed_image, fixed_intrinsic)
        wrist_observation = estimate_board_pose(wrist_image, wrist_intrinsic)
        for camera_name, observation in (
            ("fixed", fixed_observation),
            ("wrist", wrist_observation),
        ):
            if not _observation_is_acceptable(
                observation,
                min_corners=args.min_corners,
                max_reprojection_error_px=args.max_reprojection_error_px,
            ):
                raise RuntimeError(
                    f"Pose {pose_index} {camera_name} observation no longer meets "
                    f"the quality threshold: {observation.corner_count} corners, "
                    f"{observation.reprojection_rmse_px:.3f}px."
                )

        base_from_end_effector.append(
            _validated_transform(
                entry["base_from_end_effector"],
                f"pose {pose_index} base_from_end_effector",
            )
        )
        fixed_camera_from_board.append(fixed_observation.camera_from_board)
        wrist_camera_from_board.append(wrist_observation.camera_from_board)
        fixed_reprojection_errors.append(fixed_observation.reprojection_rmse_px)
        wrist_reprojection_errors.append(wrist_observation.reprojection_rmse_px)
        _save_image(
            visualization_dir / f"pose_{pose_index:02d}_fixed.png",
            _annotate_observation(
                fixed_image,
                fixed_observation,
                fixed_intrinsic,
                "fixed",
            ),
        )
        _save_image(
            visualization_dir / f"pose_{pose_index:02d}_wrist.png",
            _annotate_observation(
                wrist_image,
                wrist_observation,
                wrist_intrinsic,
                "wrist",
            ),
        )

    motion_translation_span_m, motion_rotation_span_deg = _motion_spans(
        base_from_end_effector
    )
    if motion_rotation_span_deg < 10.0:
        raise RuntimeError(
            f"Robot pose rotation span is only {motion_rotation_span_deg:.2f} deg; "
            "the dataset is degenerate for hand-eye calibration."
        )

    end_effector_from_wrist_camera = _solve_hand_eye(
        base_from_end_effector,
        wrist_camera_from_board,
        args.method,
    )
    wrist_camera_from_end_effector = _invert_transform(
        end_effector_from_wrist_camera
    )

    # A correct hand-eye transform makes base_from_board constant across poses.
    base_from_board_per_pose = [
        base_from_ee @ end_effector_from_wrist_camera @ wrist_from_board
        for base_from_ee, wrist_from_board in zip(
            base_from_end_effector,
            wrist_camera_from_board,
        )
    ]
    base_from_board = _average_transforms(base_from_board_per_pose)
    board_translation_errors, board_rotation_errors = _transform_errors(
        base_from_board,
        base_from_board_per_pose,
    )

    fixed_camera_from_board_mean = _average_transforms(fixed_camera_from_board)
    fixed_translation_errors, fixed_rotation_errors = _transform_errors(
        fixed_camera_from_board_mean,
        fixed_camera_from_board,
    )
    fixed_camera_from_base = (
        fixed_camera_from_board_mean @ _invert_transform(base_from_board)
    )
    base_from_fixed_camera = _invert_transform(fixed_camera_from_base)

    # Compare observed fixed_from_wrist against the transform predicted from
    # robot pose plus the solved hand-eye result.
    paired_translation_errors = []
    paired_rotation_errors = []
    for base_from_ee, fixed_from_board, wrist_from_board in zip(
        base_from_end_effector,
        fixed_camera_from_board,
        wrist_camera_from_board,
    ):
        observed_fixed_from_wrist = fixed_from_board @ _invert_transform(
            wrist_from_board
        )
        predicted_fixed_from_wrist = (
            fixed_camera_from_base
            @ base_from_ee
            @ end_effector_from_wrist_camera
        )
        delta = _invert_transform(observed_fixed_from_wrist) @ (
            predicted_fixed_from_wrist
        )
        paired_translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        paired_rotation_errors.append(_rotation_angle_deg(delta[:3, :3]))

    target_from_world = np.eye(4, dtype=np.float64)
    target_from_world[:3, :3] = WORLD_TO_NATIVE_BOARD
    fixed_camera_from_world = fixed_camera_from_board_mean @ target_from_world
    base_from_world = base_from_board @ target_from_world

    metrics = {
        "observation_count": len(entries),
        "motion_translation_span_m": motion_translation_span_m,
        "motion_rotation_span_deg": motion_rotation_span_deg,
        "fixed_reprojection_rmse_px_mean": float(
            np.mean(fixed_reprojection_errors)
        ),
        "fixed_reprojection_rmse_px_max": float(
            np.max(fixed_reprojection_errors)
        ),
        "wrist_reprojection_rmse_px_mean": float(
            np.mean(wrist_reprojection_errors)
        ),
        "wrist_reprojection_rmse_px_max": float(
            np.max(wrist_reprojection_errors)
        ),
        "board_in_base_translation_rmse_m": _rmse(board_translation_errors),
        "board_in_base_rotation_rmse_deg": _rmse(board_rotation_errors),
        "fixed_camera_stability_translation_rmse_m": _rmse(
            fixed_translation_errors
        ),
        "fixed_camera_stability_rotation_rmse_deg": _rmse(
            fixed_rotation_errors
        ),
        "paired_camera_translation_rmse_m": _rmse(
            np.asarray(paired_translation_errors)
        ),
        "paired_camera_rotation_rmse_deg": _rmse(
            np.asarray(paired_rotation_errors)
        ),
    }
    quality_checks = [
        _quality_check(
            "board_in_base_translation_rmse_m",
            metrics["board_in_base_translation_rmse_m"],
            0.015,
        ),
        _quality_check(
            "board_in_base_rotation_rmse_deg",
            metrics["board_in_base_rotation_rmse_deg"],
            2.0,
        ),
        _quality_check(
            "fixed_camera_stability_translation_rmse_m",
            metrics["fixed_camera_stability_translation_rmse_m"],
            0.010,
        ),
        _quality_check(
            "fixed_camera_stability_rotation_rmse_deg",
            metrics["fixed_camera_stability_rotation_rmse_deg"],
            1.0,
        ),
        _quality_check(
            "paired_camera_translation_rmse_m",
            metrics["paired_camera_translation_rmse_m"],
            0.015,
        ),
        _quality_check(
            "paired_camera_rotation_rmse_deg",
            metrics["paired_camera_rotation_rmse_deg"],
            2.0,
        ),
    ]
    quality_passed = all(check["passed"] for check in quality_checks)

    result: dict[str, Any] = {
        "schema_version": 1,
        "solved_utc": _utc_now(),
        "source_manifest": str(manifest_path),
        "method": args.method,
        "fixed_serial": manifest["fixed_serial"],
        "wrist_serial": manifest["wrist_serial"],
        "frame_conventions": {
            "end_effector_from_wrist_camera": (
                "maps wrist-camera coordinates into the measured xArm TCP frame"
            ),
            "fixed_camera_from_base": (
                "maps xArm base coordinates into the fixed-camera frame"
            ),
            "fixed_camera_from_world": (
                "maps calibrator.py flipped-board world coordinates into the "
                "fixed-camera frame"
            ),
        },
        "fixed_intrinsic": fixed_intrinsic,
        "wrist_intrinsic": wrist_intrinsic,
        "end_effector_from_wrist_camera": end_effector_from_wrist_camera,
        "wrist_camera_from_end_effector": wrist_camera_from_end_effector,
        "base_from_board_native": base_from_board,
        "base_from_world": base_from_world,
        "fixed_camera_from_board_native": fixed_camera_from_board_mean,
        "fixed_camera_from_world": fixed_camera_from_world,
        "fixed_camera_from_base": fixed_camera_from_base,
        "base_from_fixed_camera": base_from_fixed_camera,
        "metrics": metrics,
        "quality_checks": quality_checks,
        "quality_passed": quality_passed,
        "cam_params_compatible": {
            str(manifest["fixed_serial"]): {
                "extrinsic": fixed_camera_from_world,
                "intrinsic": fixed_intrinsic,
            }
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_json_path = args.output_dir / "hand_eye_report.json"
    result_pickle_path = args.output_dir / "hand_eye_params.pkl"
    _write_json_atomic(result_json_path, result)
    with result_pickle_path.open("wb") as result_file:
        pickle.dump(result, result_file)

    print("Hand-eye calibration complete.")
    print("end_effector_from_wrist_camera:")
    print(end_effector_from_wrist_camera)
    print(json.dumps(metrics, indent=2))
    print(f"Quality passed: {quality_passed}")
    print(f"Saved report: {result_json_path}")
    print(f"Saved parameters: {result_pickle_path}")
    if not quality_passed:
        for check in quality_checks:
            if not check["passed"]:
                print(
                    f"WARNING: {check['name']}={check['value']:.6g} exceeds "
                    f"{check['maximum']:.6g}.",
                    file=sys.stderr,
                )
        if args.strict_quality:
            raise RuntimeError(
                "Calibration results were saved, but strict quality checks failed."
            )
    return result


def run_synthetic_self_test() -> None:
    """Verify transform directions and OpenCV hand-eye usage without hardware."""
    end_effector_from_camera_expected = _make_transform(
        _rpy_to_rotation(0.12, -0.08, 0.21),
        np.array([0.045, -0.012, 0.082]),
    )
    base_from_board = _make_transform(
        _rpy_to_rotation(-0.15, 0.10, -0.30),
        np.array([0.55, 0.08, 0.12]),
    )
    base_from_end_effector = []
    camera_from_board = []
    for index in range(20):
        phase = index / 19.0
        base_from_ee = _make_transform(
            _rpy_to_rotation(
                -0.35 + 0.65 * phase,
                0.22 * math.sin(index * 0.7),
                -0.45 + 0.9 * phase,
            ),
            np.array(
                [
                    0.30 + 0.08 * math.sin(index * 0.5),
                    -0.12 + 0.10 * math.cos(index * 0.4),
                    0.35 + 0.04 * math.sin(index * 0.9),
                ]
            ),
        )
        camera_from_target = (
            _invert_transform(end_effector_from_camera_expected)
            @ _invert_transform(base_from_ee)
            @ base_from_board
        )
        base_from_end_effector.append(base_from_ee)
        camera_from_board.append(camera_from_target)

    solved = _solve_hand_eye(
        base_from_end_effector,
        camera_from_board,
        "park",
    )
    delta = _invert_transform(end_effector_from_camera_expected) @ solved
    translation_error = float(np.linalg.norm(delta[:3, 3]))
    rotation_error = _rotation_angle_deg(delta[:3, :3])
    if translation_error > 1e-6 or rotation_error > 1e-4:
        raise AssertionError(
            "Synthetic hand-eye self-test failed: "
            f"translation={translation_error:.3e}m, rotation={rotation_error:.3e}deg"
        )
    print(
        "Synthetic hand-eye self-test passed: "
        f"translation={translation_error:.3e}m, rotation={rotation_error:.3e}deg"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture 20 paired fixed/wrist RealSense images while moving an xArm7, "
            "then solve the wrist-camera hand-eye transform."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("capture", "solve", "all", "self-test"),
        default="all",
        help="Operation to perform (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Camera serial configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Capture and result directory (default: %(default)s).",
    )
    parser.add_argument("--robot-ip", default="192.168.1.196")
    parser.add_argument("--fixed-serial", help="Override config.yaml D455 serial.")
    parser.add_argument("--wrist-serial", help="Override config.yaml D405 serial.")
    parser.add_argument(
        "--method",
        choices=tuple(HAND_EYE_METHODS),
        default="park",
        help="OpenCV hand-eye method used during solve (default: %(default)s).",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=35,
        help="Minimum ChArUco corners required in each image (default: %(default)s).",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=0.5,
        help="Maximum per-image reprojection RMSE (default: %(default)s).",
    )
    parser.add_argument(
        "--capture-attempts",
        type=int,
        default=20,
        help="Maximum frame attempts at each robot pose (default: %(default)s).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.75,
        help="Delay after each completed robot move (default: %(default)s).",
    )
    parser.add_argument(
        "--joint-speed-deg-s",
        type=float,
        default=15.0,
        help="Robot joint-motion speed in degrees/second (default: %(default)s).",
    )
    parser.add_argument(
        "--show-gui",
        action="store_true",
        help="Show annotated fixed/wrist images during capture.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the MOVE prompt after independently verifying all robot poses.",
    )
    parser.add_argument(
        "--overwrite-captures",
        action="store_true",
        help="Allow replacement of the existing capture manifest and pose images.",
    )
    parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Return an error after saving results when quality checks fail.",
    )
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.min_corners < 4:
        parser.error("--min-corners must be at least 4")
    if args.max_reprojection_error_px <= 0.0:
        parser.error("--max-reprojection-error-px must be positive")
    if args.capture_attempts < 1:
        parser.error("--capture-attempts must be at least 1")
    if args.settle_seconds < 0.0:
        parser.error("--settle-seconds cannot be negative")
    if args.joint_speed_deg_s <= 0.0:
        parser.error("--joint-speed-deg-s must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "self-test":
        run_synthetic_self_test()
        return
    if args.mode in ("capture", "all"):
        capture_dataset(args)
    if args.mode in ("solve", "all"):
        solve_dataset(args)


if __name__ == "__main__":
    main()
