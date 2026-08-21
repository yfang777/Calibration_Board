"""Calibrate a wrist-mounted RealSense camera against an xArm end effector.

This is the project-local variant of ``ref.py``.  It keeps the reference
workflow and its xArm/RealSense wrappers while using this repository's
ChArUco board, capture poses, and image-quality requirements.

Typical use from the repository root in the existing robot environment::

    python debug/hand_eye_calibrate/calibrator_hand_eye.py \
        debug/hand_eye_calibrate/data --mode all

The calibration board must remain fixed for the complete run.  For a split
run, leave the robot at ``CAPTURE_JOINTS_DEG[0]`` while running ``fixed``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from scipy.spatial.transform import Rotation

from robot_control.src.robot_control.calibration import calibration as calib
from robot_control.src.robot_control.calibration.ee_transformation import XArmRobot
from robot_control.src.robot_control.calibration.realsense_manager import (
    RealSenseManager,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

FRAME_WIDTH = 848
FRAME_HEIGHT = 480
FRAME_FPS = 30
COLOR_FRAME_FORMAT = rs.format.bgr8
DEPTH_FRAME_FORMAT = rs.format.z16

BOARD_DICTIONARY_NAME = "DICT_5X5_250"
BOARD_SIZE = (12, 9)
BOARD_SQUARE_M = 0.03
BOARD_MARKER_M = 0.022
DISTORTION = np.zeros(5, dtype=np.float64)
WORLD_TO_NATIVE_BOARD = np.diag([1.0, -1.0, -1.0])

D455_MIN_CORNERS = 25
D405_MIN_CORNERS = 25
D455_MAX_REPROJECTION_ERROR_PX = 0.25
D405_MAX_REPROJECTION_ERROR_PX = 0.5

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
        # [0.1, 14.1, 0.3, 88.3, 0.5, 95.7, -0.1],
        # [0.1, 19.1, 0.4, 96.1, 0.8, 109.9, 0.0],
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

POSE_INDEX_PATTERN = re.compile(r"(?:@pose_|_)(\d+)$")


@dataclass(frozen=True)
class BoardObservation:
    camera_from_native_board: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: Optional[np.ndarray]
    reprojection_rmse_px: float

    @property
    def corner_count(self) -> int:
        return len(self.charuco_ids)


def _deg_to_rad(joints_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(joints_deg)


def _compose_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _native_from_world_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = WORLD_TO_NATIVE_BOARD
    return transform


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "model": str(intrinsics.model),
        "coeffs": [float(coefficient) for coefficient in intrinsics.coeffs],
    }


def _intrinsic_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _load_serials(
    config_path: Path,
    front_override: Optional[str],
    wrist_override: Optional[str],
) -> dict[str, str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_front = next(iter(config["rs_id"]["d455"].values()))
    configured_wrist = next(iter(config["rs_id"]["d405"].values()))
    return {
        "front": str(front_override or configured_front),
        "wrist": str(wrist_override or configured_wrist),
    }


def _camera_quality(camera_name: str) -> tuple[int, float]:
    if camera_name == "front":
        return D455_MIN_CORNERS, D455_MAX_REPROJECTION_ERROR_PX
    if camera_name == "wrist":
        return D405_MIN_CORNERS, D405_MAX_REPROJECTION_ERROR_PX
    raise ValueError(f"Unknown camera name: {camera_name}")


def estimate_board_pose(
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
) -> BoardObservation:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        CHARUCO_DETECTOR.detectBoard(gray)
    )
    corner_count = 0 if charuco_corners is None else len(charuco_corners)
    if charuco_ids is None or corner_count < 4:
        raise RuntimeError(
            f"ChArUco detection found {corner_count} corners; "
            "at least four are required."
        )

    ids = charuco_ids.reshape(-1).astype(np.int64)
    board_points = np.asarray(
        CALIBRATION_BOARD.getChessboardCorners(), dtype=np.float32
    )
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

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsic,
        DISTORTION,
    )
    residuals = projected_points.reshape(-1, 2) - image_points.reshape(-1, 2)
    reprojection_rmse_px = float(
        np.sqrt(np.mean(np.sum(residuals * residuals, axis=1)))
    )
    rotation, _ = cv2.Rodrigues(rvec)
    return BoardObservation(
        camera_from_native_board=_compose_transform(rotation, tvec),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32),
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32),
        marker_corners=tuple(marker_corners),
        marker_ids=marker_ids,
        reprojection_rmse_px=reprojection_rmse_px,
    )


def _quality_error(
    camera_name: str,
    observation: BoardObservation,
) -> Optional[str]:
    min_corners, max_reprojection_error_px = _camera_quality(camera_name)
    if observation.corner_count < min_corners:
        return (
            f"{camera_name} found {observation.corner_count} ChArUco corners; "
            f"at least {min_corners} are required"
        )
    if observation.reprojection_rmse_px > max_reprojection_error_px:
        return (
            f"{camera_name} reprojection RMSE is "
            f"{observation.reprojection_rmse_px:.3f}px; maximum is "
            f"{max_reprojection_error_px:.3f}px"
        )
    return None


def _annotate_observation(
    image_bgr: np.ndarray,
    intrinsic: np.ndarray,
    camera_name: str,
    observation: BoardObservation,
) -> np.ndarray:
    annotated = image_bgr.copy()
    if observation.marker_corners:
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
    label = (
        f"{camera_name}: {observation.corner_count} corners, "
        f"{observation.reprojection_rmse_px:.3f}px"
    )
    cv2.putText(
        annotated,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _capture_accepted_observation(
    camera: RealSenseManager,
    camera_name: str,
    intrinsic: np.ndarray,
    attempts: int,
) -> tuple[np.ndarray, BoardObservation]:
    last_error = "no frame received"
    for attempt in range(1, attempts + 1):
        camera.poll_frames()
        image = camera.get_color_image().copy()
        try:
            observation = estimate_board_pose(image, intrinsic)
        except RuntimeError as error:
            last_error = str(error)
            print(f"  {camera_name} attempt {attempt}/{attempts}: {last_error}")
            continue
        last_error = _quality_error(camera_name, observation)
        if last_error is None:
            return image, observation
        print(f"  {camera_name} attempt {attempt}/{attempts}: {last_error}")
    raise RuntimeError(
        f"Could not capture an acceptable {camera_name} observation after "
        f"{attempts} attempts: {last_error}"
    )


def _pose_index(path: Path) -> int:
    match = POSE_INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"File does not contain a pose index: {path}")
    return int(match.group(1))


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _handeye_consistency(
    end_effector_from_camera: np.ndarray,
    base_from_end_effector: list[np.ndarray],
    camera_from_board: list[np.ndarray],
) -> dict[str, float]:
    base_from_board = [
        base_from_ee @ end_effector_from_camera @ camera_from_target
        for base_from_ee, camera_from_target in zip(
            base_from_end_effector,
            camera_from_board,
        )
    ]
    translations = np.asarray(
        [transform[:3, 3] for transform in base_from_board], dtype=np.float64
    )
    reference_rotation = base_from_board[0][:3, :3]
    rotation_errors_deg = np.asarray(
        [
            _rotation_angle_deg(reference_rotation.T @ transform[:3, :3])
            for transform in base_from_board
        ]
    )
    centered_translations = translations - translations.mean(axis=0)
    return {
        "observation_count": len(base_from_board),
        "board_translation_rmse_m": float(
            np.sqrt(np.mean(np.sum(centered_translations**2, axis=1)))
        ),
        "board_rotation_rmse_deg": float(
            np.sqrt(np.mean(rotation_errors_deg**2))
        ),
    }


@dataclass
class HandEyeCalibrator:
    work_dir: Path
    robot_ip: Optional[str]
    show_gui: bool = False
    init_robot: bool = True
    capture_attempts: int = 20

    robot: Optional[XArmRobot] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.work_dir = self.work_dir.expanduser().resolve()
        image_root = self.work_dir / "calibration_images"
        self.handeye_images_dir = image_root / "handeye"
        self.compose_images_dir = image_root / "compose"
        self.marked_images_dir = image_root / "marked"
        self.handeye_images_dir.mkdir(parents=True, exist_ok=True)
        self.compose_images_dir.mkdir(parents=True, exist_ok=True)
        self.marked_images_dir.mkdir(parents=True, exist_ok=True)
        self.extrinsics_path = self.work_dir / "extrinsics.json"
        self.intrinsics_path = self.work_dir / "intrinsics.json"

        if self.init_robot:
            if self.robot_ip is None:
                raise ValueError("robot_ip is required when init_robot is true")
            self.robot = XArmRobot(
                ip=self.robot_ip,
                control_frequency=500.0,
                max_delta=0.01,
            )

    def fixed_calibration(self, serials: dict[str, str]) -> None:
        """Capture one board observation and active color intrinsics per camera."""
        extrinsics = _read_json(self.extrinsics_path)
        intrinsics = _read_json(self.intrinsics_path)
        extrinsics["board"] = {
            "dictionary": BOARD_DICTIONARY_NAME,
            "size": BOARD_SIZE,
            "square_length_m": BOARD_SQUARE_M,
            "marker_length_m": BOARD_MARKER_M,
            "world_to_native_board_rotation": WORLD_TO_NATIVE_BOARD,
        }
        extrinsics["frame_conventions"] = {
            "board2cam": "camera_from_native_board",
            "world2cam": "camera_from_flipped_board_world",
            "wristcam2ee": "end_effector_from_wrist_camera",
            "ee2base": "base_from_end_effector_at_pose_0",
        }
        board_to_camera = extrinsics.setdefault("board2cam", {})
        world_to_camera = extrinsics.setdefault("world2cam", {})
        native_from_world = _native_from_world_transform()

        if self.robot is not None:
            self.robot.move_and_wait_robot(_deg_to_rad(CAPTURE_JOINTS_DEG[0]))

        for camera_name, serial in serials.items():
            print(f"Capturing fixed calibration for {camera_name} ({serial})...")
            camera = RealSenseManager(
                serial=serial,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
                fps=FRAME_FPS,
                depth_format=DEPTH_FRAME_FORMAT,
                color_format=COLOR_FRAME_FORMAT,
            )
            try:
                intrinsic_data = _intrinsics_to_dict(camera.get_color_intrinsic())
                intrinsic = _intrinsic_matrix(intrinsic_data)
                image, observation = _capture_accepted_observation(
                    camera,
                    camera_name,
                    intrinsic,
                    self.capture_attempts,
                )
            finally:
                camera.stop()

            board_to_camera[camera_name] = observation.camera_from_native_board
            world_to_camera[camera_name] = (
                observation.camera_from_native_board @ native_from_world
            )
            intrinsics[camera_name] = intrinsic_data
            cv2.imwrite(
                str(self.compose_images_dir / f"{camera_name}_image@pose_0.png"),
                image,
            )
            annotated = _annotate_observation(
                image,
                intrinsic,
                camera_name,
                observation,
            )
            cv2.imwrite(
                str(self.marked_images_dir / f"{camera_name}_fixed_calibration.png"),
                annotated,
            )
            if self.show_gui:
                cv2.imshow(f"{camera_name} fixed calibration", annotated)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

            print(
                f"  accepted {observation.corner_count} corners at "
                f"{observation.reprojection_rmse_px:.3f}px RMSE"
            )
            _write_json(self.extrinsics_path, extrinsics)
            _write_json(self.intrinsics_path, intrinsics)

    def capture_wrist_images(self, wrist_serial: str) -> None:
        """Move through the configured sweep and save synchronized image/EE pairs."""
        if self.robot is None:
            raise RuntimeError("Wrist capture requires an initialized robot.")
        intrinsics = _read_json(self.intrinsics_path)
        wrist_intrinsic = _intrinsic_matrix(intrinsics["wrist"])

        print("Moving the robot to the initial wrist-capture pose...")
        self.robot.move_and_wait_robot(_deg_to_rad(CAPTURE_JOINTS_DEG[0]))
        camera = RealSenseManager(
            serial=wrist_serial,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=FRAME_FPS,
            depth_format=DEPTH_FRAME_FORMAT,
            color_format=COLOR_FRAME_FORMAT,
        )
        try:
            for pose_index, joints_deg in enumerate(CAPTURE_JOINTS_DEG):
                print(
                    f"Capturing pose {pose_index + 1}/{len(CAPTURE_JOINTS_DEG)}: "
                    f"{joints_deg.tolist()}"
                )
                self.robot.move_and_wait_robot(_deg_to_rad(joints_deg))
                image, observation = _capture_accepted_observation(
                    camera,
                    "wrist",
                    wrist_intrinsic,
                    self.capture_attempts,
                )
                robot_state = self.robot.get_state()
                ee_data = {
                    "translation": robot_state.cartesian_pos(),
                    "quat": robot_state.quat(),
                    "joints_rad": robot_state.joints(),
                    "commanded_joints_deg": joints_deg,
                    "corner_count": observation.corner_count,
                    "reprojection_rmse_px": observation.reprojection_rmse_px,
                }

                image_path = (
                    self.handeye_images_dir
                    / f"wrist_image@pose_{pose_index}.png"
                )
                pose_path = (
                    self.handeye_images_dir
                    / f"wrist_ee_pose_{pose_index}.json"
                )
                cv2.imwrite(str(image_path), image)
                _write_json(pose_path, ee_data)
                if pose_index == 0:
                    cv2.imwrite(
                        str(self.compose_images_dir / "wrist_image@pose_0.png"),
                        image,
                    )

                annotated = _annotate_observation(
                    image,
                    wrist_intrinsic,
                    "wrist",
                    observation,
                )
                cv2.imwrite(
                    str(
                        self.marked_images_dir
                        / f"handeye_capture_{pose_index}.png"
                    ),
                    annotated,
                )
                if self.show_gui:
                    cv2.imshow(f"Wrist pose {pose_index}", annotated)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
        finally:
            print("Returning the robot to the initial pose...")
            self.robot.move_and_wait_robot(_deg_to_rad(CAPTURE_JOINTS_DEG[0]))
            camera.stop()

    def handeye_calibration(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve camera-to-EE from previously captured wrist observations."""
        extrinsics = _read_json(self.extrinsics_path)
        intrinsics = _read_json(self.intrinsics_path)
        wrist_intrinsic = _intrinsic_matrix(intrinsics["wrist"])

        image_paths = {
            _pose_index(path): path
            for path in self.handeye_images_dir.glob("wrist_image@pose_*.png")
        }
        pose_paths = {
            _pose_index(path): path
            for path in self.handeye_images_dir.glob("wrist_ee_pose_*.json")
        }
        if image_paths.keys() != pose_paths.keys():
            raise RuntimeError(
                "Wrist image and end-effector pose indices do not match: "
                f"images={sorted(image_paths)}, poses={sorted(pose_paths)}"
            )
        expected_pose_indices = set(range(len(CAPTURE_JOINTS_DEG)))
        if image_paths.keys() != expected_pose_indices:
            raise RuntimeError(
                f"Expected wrist poses {sorted(expected_pose_indices)}, "
                f"found {sorted(image_paths)}."
            )

        base_from_end_effector: list[np.ndarray] = []
        camera_from_board: list[np.ndarray] = []
        first_observation: Optional[BoardObservation] = None
        for pose_index in sorted(image_paths):
            image = cv2.imread(str(image_paths[pose_index]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {image_paths[pose_index]}")
            observation = estimate_board_pose(image, wrist_intrinsic)
            quality_error = _quality_error("wrist", observation)
            if quality_error is not None:
                raise RuntimeError(f"Pose {pose_index}: {quality_error}")

            ee_data = json.loads(pose_paths[pose_index].read_text(encoding="utf-8"))
            rotation = Rotation.from_quat(ee_data["quat"]).as_matrix()
            base_from_end_effector.append(
                _compose_transform(rotation, ee_data["translation"])
            )
            camera_from_board.append(observation.camera_from_native_board)
            if first_observation is None:
                first_observation = observation

            annotated = _annotate_observation(
                image,
                wrist_intrinsic,
                "wrist",
                observation,
            )
            cv2.imwrite(
                str(
                    self.marked_images_dir
                    / f"handeye_calibration_{pose_index}.png"
                ),
                annotated,
            )

        rotations_end_effector_to_base = [
            transform[:3, :3] for transform in base_from_end_effector
        ]
        translations_end_effector_to_base = [
            transform[:3, 3] for transform in base_from_end_effector
        ]
        rotations_board_to_camera = [
            transform[:3, :3] for transform in camera_from_board
        ]
        translations_board_to_camera = [
            transform[:3, 3].reshape(3, 1) for transform in camera_from_board
        ]
        end_effector_from_camera = calib.solve_EE_to_camera(
            rotations_end_effector_to_base,
            translations_end_effector_to_base,
            rotations_board_to_camera,
            translations_board_to_camera,
        )
        base_from_end_effector_at_pose_0 = base_from_end_effector[0]

        extrinsics["wristcam2ee"] = end_effector_from_camera
        extrinsics["ee2base"] = base_from_end_effector_at_pose_0
        extrinsics["handeye_metrics"] = _handeye_consistency(
            end_effector_from_camera,
            base_from_end_effector,
            camera_from_board,
        )
        extrinsics.setdefault("board2cam", {})["wrist"] = (
            first_observation.camera_from_native_board
        )
        extrinsics.setdefault("world2cam", {})["wrist"] = (
            first_observation.camera_from_native_board
            @ _native_from_world_transform()
        )
        _write_json(self.extrinsics_path, extrinsics)

        print("Hand-eye calibration complete. end_effector_from_wrist_camera:")
        print(end_effector_from_camera)
        print(json.dumps(extrinsics["handeye_metrics"], indent=2))
        return end_effector_from_camera, base_from_end_effector_at_pose_0

    def compose(self) -> None:
        """Compose wrist/front/base transforms at capture pose zero."""
        extrinsics = _read_json(self.extrinsics_path)
        board_to_camera = extrinsics["board2cam"]
        end_effector_from_wrist = np.asarray(
            extrinsics["wristcam2ee"], dtype=np.float64
        )
        base_from_end_effector = np.asarray(extrinsics["ee2base"], dtype=np.float64)
        wrist_from_board = np.asarray(board_to_camera["wrist"], dtype=np.float64)
        front_from_board = np.asarray(board_to_camera["front"], dtype=np.float64)

        base_from_wrist = base_from_end_effector @ end_effector_from_wrist
        wrist_from_front = wrist_from_board @ _invert_transform(front_from_board)
        front_from_wrist = _invert_transform(wrist_from_front)
        base_from_front = base_from_wrist @ wrist_from_front
        base_from_native_board = base_from_wrist @ wrist_from_board
        base_from_world = base_from_native_board @ _native_from_world_transform()

        extrinsics["cam2cam"] = {
            "front2wrist": wrist_from_front,
            "wrist2front": front_from_wrist,
        }
        extrinsics["cam2base"] = {
            "wrist2base": base_from_wrist,
            "front2base": base_from_front,
        }
        extrinsics["board2base"] = base_from_native_board
        extrinsics["world2base"] = base_from_world
        _write_json(self.extrinsics_path, extrinsics)
        print("Composed front/wrist/base transforms in extrinsics.json.")

    def close(self) -> None:
        cv2.destroyAllWindows()
        if self.robot is not None:
            self.robot.stop()


def _hardware_reset(serials: dict[str, str]) -> None:
    requested_serials = set(serials.values())
    context = rs.context()
    connected = {
        device.get_info(rs.camera_info.serial_number): device
        for device in context.devices
    }
    missing = requested_serials - connected.keys()
    if missing:
        raise RuntimeError(f"RealSense cameras are not connected: {sorted(missing)}")
    for serial in requested_serials:
        device = connected[serial]
        name = device.get_info(rs.camera_info.name)
        print(f"Hardware-resetting {name} ({serial})...")
        device.hardware_reset()
    del context
    time.sleep(3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate fixed and wrist RealSense cameras against an xArm."
    )
    parser.add_argument("work_dir", type=Path)
    parser.add_argument(
        "--mode",
        choices=("fixed", "handeye", "compose", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--robot-ip", default="192.168.1.196")
    parser.add_argument("--front-serial", help="Override the configured D455 serial.")
    parser.add_argument("--wrist-serial", help="Override the configured D405 serial.")
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--capture-attempts", type=int, default=20)
    parser.add_argument("--skip-hardware-reset", action="store_true")
    args = parser.parse_args()
    if args.capture_attempts < 1:
        parser.error("--capture-attempts must be at least one")
    return args


def main() -> None:
    args = parse_args()
    serials = _load_serials(
        args.config.expanduser().resolve(),
        args.front_serial,
        args.wrist_serial,
    )
    if args.mode in ("fixed", "handeye", "all") and not args.skip_hardware_reset:
        _hardware_reset(serials)

    init_robot = args.mode in ("handeye", "all")
    calibrator = HandEyeCalibrator(
        work_dir=args.work_dir,
        robot_ip=args.robot_ip if init_robot else None,
        show_gui=args.show_gui,
        init_robot=init_robot,
        capture_attempts=args.capture_attempts,
    )
    try:
        if args.mode in ("fixed", "all"):
            calibrator.fixed_calibration(serials)
        if args.mode in ("handeye", "all"):
            calibrator.capture_wrist_images(serials["wrist"])
            calibrator.handeye_calibration()
        if args.mode in ("compose", "all"):
            calibrator.compose()
    finally:
        calibrator.close()


if __name__ == "__main__":
    main()
