"""Calibrate the D405 pose and compare it with a matching MuJoCo view.

The physical robot must be at ``ROBOT_JOINTS_DEG`` and the ChArUco board must
remain fixed at the calibration-board world origin. The script first estimates
the D405's world pose from several ChArUco detections. It then converts that
world pose into the wrist camera's parent-body frame in MuJoCo and displays:

    MuJoCo render | D405 image | 50/50 overlay

Pass ``--is-warp`` to map the D405 image into MuJoCo's intrinsic frame before
blending. This uses only the two cameras' intrinsic matrices; it does not fit a
transform from ChArUco correspondences.

The calibrated pose is applied to the in-memory MuJoCo model only; this script
does not modify xarm7_with_wrist.xml.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np


# Support direct execution as well as `python -m debug.tmp_wrist_calibrate`.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from digital_twin_camera import utils  # noqa: E402
from digital_twin_camera.mujoco_sync import (  # noqa: E402
    H,
    SCENE_XML,
    W,
    WRIST_CAMERA_NAME,
    WRIST_ROBOT_XML,
    RealSenseRgbCameras,
    assert_realsense_connected,
    get_mj_cam,
    load_camera_serials,
    warp_image,
    write_scene_xml,
)


ROBOT_JOINTS_DEG = (0.2, -32.0, 0.0, 56.7, 0.2, 86.5, 0.0)
GRIPPER_JOINT_NAMES = (
    "left_driver_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_driver_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
GRIPPER_CLOSED_QPOS = 0.0
BOARD_SIZE = (12, 9)
BOARD_SQUARE_M = 0.03
BOARD_MARKER_M = 0.022
CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])

CALIBRATION_WINDOW = "D405 ChArUco wrist calibration"
COMPARISON_WINDOW = "Calibrated wrist camera: MuJoCo | D405 | overlay"


@dataclass(frozen=True)
class D405PoseCalibration:
    """D405 pinhole calibration in the calibration-board world frame."""

    serial: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    world_to_camera: np.ndarray
    reprojection_error_px: float
    detected_corners: int
    sample_count: int

    @property
    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera)

    @property
    def position_world(self) -> np.ndarray:
        return self.camera_to_world[:3, 3]


class CalibrationAborted(RuntimeError):
    """Raised when the user closes or cancels the calibration window."""


def _make_charuco_detector() -> tuple[
    cv2.aruco.CharucoBoard,
    cv2.aruco.CharucoDetector,
]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard(
        size=BOARD_SIZE,
        squareLength=BOARD_SQUARE_M,
        markerLength=BOARD_MARKER_M,
        dictionary=dictionary,
    )
    detector = cv2.aruco.CharucoDetector(
        board,
        cv2.aruco.CharucoParameters(),
        cv2.aruco.DetectorParameters(),
    )
    return board, detector


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    """Project the arithmetic mean of rotation matrices back onto SO(3)."""
    mean_rotation = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, color, 1, cv2.LINE_AA)


def _window_was_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1.0
    except cv2.error:
        return True


def calibrate_d405_pose(
    camera: RealSenseRgbCameras,
    serial: str,
    *,
    sample_count: int = 10,
    min_corners: int = 35,
    max_reprojection_error_px: float = 0.25,
    visualize: bool = True,
) -> D405PoseCalibration:
    """Estimate the D405 world pose from repeated ChArUco detections.

    ``camera`` must already be started. The returned extrinsic maps points from
    the repository's calibration-board world frame into OpenCV camera space.
    The world convention matches ``calibrator.py`` and ``mujoco_sync.py``:
    world X equals board X while world Y and Z are flipped from the board's
    native ChArUco frame.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    if min_corners < 4:
        raise ValueError("min_corners must be at least 4.")
    if max_reprojection_error_px <= 0.0:
        raise ValueError("max_reprojection_error_px must be positive.")
    if serial not in camera.intrinsics:
        raise RuntimeError(f"RealSense {serial} is not started.")

    intrinsic = camera.intrinsics[serial].copy()
    # The configured RealSense color stream is rectified. Keep the same zero-
    # distortion convention used by calibrator.py and by the MuJoCo pinhole camera.
    distortion = np.zeros(5, dtype=np.float64)
    board, detector = _make_charuco_detector()
    chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)

    rotations_world_to_camera: list[np.ndarray] = []
    translations_world_to_camera: list[np.ndarray] = []
    accepted_errors: list[float] = []
    accepted_corner_counts: list[int] = []

    if visualize:
        cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CALIBRATION_WINDOW, W, H)

    try:
        while len(accepted_errors) < sample_count:
            color_bgr = camera.get_rgb()[serial]
            charuco_corners, charuco_ids, marker_corners, marker_ids = (
                detector.detectBoard(color_bgr)
            )
            detected_count = 0 if charuco_corners is None else len(charuco_corners)
            accepted = False
            error_px = float("inf")
            raw_rvec = None
            raw_tvec = None

            if charuco_ids is not None and detected_count >= min_corners:
                ids = charuco_ids.reshape(-1).astype(np.int64)
                object_points = np.ascontiguousarray(chessboard_corners[ids])
                image_points = np.ascontiguousarray(charuco_corners, dtype=np.float32)
                solved, raw_rvec, raw_tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    intrinsic,
                    distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if solved:
                    projected, _ = cv2.projectPoints(
                        object_points,
                        raw_rvec,
                        raw_tvec,
                        intrinsic,
                        distortion,
                    )
                    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
                    error_px = float(
                        np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
                    )
                    accepted = error_px <= max_reprojection_error_px

                if accepted:
                    rotation_board_to_camera = cv2.Rodrigues(raw_rvec)[0]
                    rotation_world_to_camera = (
                        rotation_board_to_camera @ CAMERA_AXIS_FLIP
                    )
                    rotations_world_to_camera.append(rotation_world_to_camera)
                    translations_world_to_camera.append(raw_tvec.reshape(3).copy())
                    accepted_errors.append(error_px)
                    accepted_corner_counts.append(detected_count)

            if visualize:
                annotated = color_bgr.copy()
                if marker_corners is not None and len(marker_corners) > 0:
                    cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
                if charuco_corners is not None and detected_count > 0:
                    cv2.aruco.drawDetectedCornersCharuco(
                        annotated, charuco_corners, charuco_ids
                    )
                if raw_rvec is not None and raw_tvec is not None:
                    cv2.drawFrameAxes(
                        annotated,
                        intrinsic,
                        distortion,
                        raw_rvec,
                        raw_tvec,
                        0.1,
                    )

                progress = f"accepted {len(accepted_errors)}/{sample_count}"
                if np.isfinite(error_px):
                    quality = (
                        f"corners {detected_count}, "
                        f"reprojection RMSE {error_px:.3f} px"
                    )
                else:
                    quality = f"corners {detected_count}, waiting for board"
                status_color = (80, 255, 80) if accepted else (0, 220, 255)
                _put_text(annotated, progress, (10, 25), 0.65, status_color)
                _put_text(annotated, quality, (10, 52), 0.55, status_color)
                _put_text(
                    annotated,
                    "Hold robot and board still | Esc cancels",
                    (10, H - 15),
                )
                cv2.imshow(CALIBRATION_WINDOW, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or _window_was_closed(CALIBRATION_WINDOW):
                    raise CalibrationAborted("D405 calibration was cancelled.")
    finally:
        if visualize:
            cv2.destroyWindow(CALIBRATION_WINDOW)

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = _average_rotations(rotations_world_to_camera)
    world_to_camera[:3, 3] = np.mean(translations_world_to_camera, axis=0)
    return D405PoseCalibration(
        serial=serial,
        intrinsic=intrinsic,
        distortion=distortion,
        world_to_camera=world_to_camera,
        reprojection_error_px=float(np.mean(accepted_errors)),
        detected_corners=int(round(float(np.mean(accepted_corner_counts)))),
        sample_count=sample_count,
    )


def set_mujoco_camera_intrinsic(
    model: mujoco.MjModel,
    camera_id: int,
    intrinsic: np.ndarray,
    image_size: tuple[int, int] = (W, H),
) -> None:
    """Apply an OpenCV intrinsic matrix to a MuJoCo camera in memory."""
    width, height = image_size
    sensor_width, sensor_height = model.cam_sensorsize[camera_id]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    model.cam_intrinsic[camera_id, 0] = fx / width * sensor_width
    model.cam_intrinsic[camera_id, 1] = fy / height * sensor_height
    model.cam_intrinsic[camera_id, 2] = (cx - width / 2.0) / width * sensor_width
    model.cam_intrinsic[camera_id, 3] = (height / 2.0 - cy) / height * sensor_height


def set_mujoco_camera_from_world_extrinsic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    world_to_camera: np.ndarray,
) -> None:
    """Set a body-mounted MuJoCo camera from an OpenCV world-to-camera pose."""
    if world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape (4, 4).")

    rotation_world_to_cv_camera = world_to_camera[:3, :3]
    translation_world_to_cv_camera = world_to_camera[:3, 3]
    camera_position_world = (
        -rotation_world_to_cv_camera.T @ translation_world_to_cv_camera
    )
    rotation_mujoco_camera_to_world = (
        rotation_world_to_cv_camera.T @ CAMERA_AXIS_FLIP
    )

    parent_body_id = int(model.cam_bodyid[camera_id])
    rotation_parent_to_world = data.xmat[parent_body_id].reshape(3, 3)
    position_parent_world = data.xpos[parent_body_id]
    camera_position_parent = rotation_parent_to_world.T @ (
        camera_position_world - position_parent_world
    )
    rotation_camera_to_parent = (
        rotation_parent_to_world.T @ rotation_mujoco_camera_to_world
    )

    camera_quat_parent = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(camera_quat_parent, rotation_camera_to_parent.reshape(-1))
    model.cam_pos[camera_id] = camera_position_parent
    model.cam_quat[camera_id] = camera_quat_parent
    mujoco.mj_forward(model, data)


def close_mujoco_gripper(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set the simulated xArm gripper qpos to fully closed."""
    for joint_name in GRIPPER_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo gripper joint {joint_name!r} was not found.")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = GRIPPER_CLOSED_QPOS

    mujoco.mj_forward(model, data)


def build_calibrated_mujoco_robot(
    calibration: D405PoseCalibration,
    joints_deg: tuple[float, ...] = ROBOT_JOINTS_DEG,
) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """Build the wrist robot at ``joints_deg`` with the calibrated D405 pose."""
    if len(joints_deg) != 7:
        raise ValueError(f"Expected 7 robot joint angles, received {len(joints_deg)}.")

    write_scene_xml(include_robot=True, robot_xml=WRIST_ROBOT_XML)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    utils.set_robot_pose(model, data, joints_deg)
    close_mujoco_gripper(model, data)

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_NAME
    )
    if camera_id < 0:
        raise RuntimeError(
            f"MuJoCo camera {WRIST_CAMERA_NAME!r} was not found in {SCENE_XML}."
        )

    set_mujoco_camera_from_world_extrinsic(
        model, data, camera_id, calibration.world_to_camera
    )
    set_mujoco_camera_intrinsic(model, camera_id, calibration.intrinsic)
    mujoco.mj_forward(model, data)
    return model, data, camera_id


def _compose_overlay_view(
    sim_bgr: np.ndarray,
    real_bgr: np.ndarray,
    calibration: D405PoseCalibration,
    *,
    is_warp: bool = False,
) -> np.ndarray:
    if sim_bgr.shape != real_bgr.shape:
        raise RuntimeError(
            "MuJoCo and D405 image shapes differ: "
            f"{sim_bgr.shape} versus {real_bgr.shape}."
        )

    sim_panel = sim_bgr.copy()
    real_panel = real_bgr.copy()
    overlay_panel = cv2.addWeighted(sim_bgr, 0.5, real_bgr, 0.5, 0.0)
    real_prefix = "Intrinsic-warped" if is_warp else "Raw"
    _put_text(sim_panel, "MuJoCo calibrated wrist_camera", (10, 25), 0.62)
    _put_text(
        real_panel,
        f"{real_prefix} D405 ({calibration.serial})",
        (10, 25),
        0.62,
    )
    _put_text(overlay_panel, "MuJoCo / D405 50/50 overlay", (10, 25), 0.62)

    panels = np.hstack((sim_panel, real_panel, overlay_panel))
    footer = np.zeros((66, panels.shape[1], 3), dtype=np.uint8)
    display = np.vstack((panels, footer))
    position = " ".join(f"{value:.5f}" for value in calibration.position_world)
    joints = " ".join(f"{value:g}" for value in ROBOT_JOINTS_DEG)
    footer_line_one = (
        f"D405 world position: [{position}] m | calibration RMSE: "
        f"{calibration.reprojection_error_px:.3f} px | "
        f"{'intrinsic warp' if is_warp else 'no warp'}"
    )
    footer_line_two = f"Joints: [{joints}] | Esc exits"
    _put_text(display, footer_line_one, (10, H + 27), 0.48)
    _put_text(display, footer_line_two, (10, H + 55), 0.49)
    return display


def visualize_real_sim_overlay(
    camera: RealSenseRgbCameras,
    calibration: D405PoseCalibration,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    is_warp: bool = False,
) -> None:
    """Show MuJoCo, D405, and their overlay with optional intrinsic warping."""
    renderer = mujoco.Renderer(model, height=H, width=W)
    try:
        mujoco_intrinsic = None
        if is_warp:
            _, mujoco_intrinsic = get_mj_cam(
                model,
                data,
                WRIST_CAMERA_NAME,
                W,
                H,
            )
        mode = "intrinsic-warped" if is_warp else "unwarped"
        print(f"Showing a 50/50 {mode} D405 and MuJoCo overlay.")

        renderer.update_scene(data, camera=WRIST_CAMERA_NAME)
        sim_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        cv2.namedWindow(COMPARISON_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(COMPARISON_WINDOW, W * 3, H + 66)
        while True:
            real_bgr = camera.get_rgb()[calibration.serial]
            if mujoco_intrinsic is not None:
                real_bgr = warp_image(
                    real_bgr,
                    calibration.intrinsic,
                    mujoco_intrinsic,
                    (W, H),
                )
            display = _compose_overlay_view(
                sim_bgr,
                real_bgr,
                calibration,
                is_warp=is_warp,
            )
            cv2.imshow(COMPARISON_WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or _window_was_closed(COMPARISON_WINDOW):
                break
    finally:
        renderer.close()
        cv2.destroyWindow(COMPARISON_WINDOW)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the configured wrist D405 against the ChArUco board, "
            "apply the result to MuJoCo at the configured robot pose, and show "
            "a direct real/sim overlay."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of accepted ChArUco frames to average (default: %(default)s).",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=35,
        help="Minimum ChArUco corners in an accepted frame (default: %(default)s).",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=0.5,
        help="Maximum accepted reprojection RMSE in pixels (default: %(default)s).",
    )
    parser.add_argument(
        "--is-warp",
        action="store_true",
        help="Warp D405 intrinsics into the MuJoCo image frame before overlaying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, wrist_serial = load_camera_serials()
    assert_realsense_connected([wrist_serial])

    print(f"Configured D405: {wrist_serial}")
    print(f"Physical and MuJoCo robot joint angles (degrees): {ROBOT_JOINTS_DEG}")
    print("Keep the physical robot and ChArUco board fixed during calibration.")

    camera = RealSenseRgbCameras([wrist_serial], wrist_rs_id=wrist_serial)
    try:
        camera.start()
        calibration = calibrate_d405_pose(
            camera,
            wrist_serial,
            sample_count=args.samples,
            min_corners=args.min_corners,
            max_reprojection_error_px=args.max_reprojection_error_px,
        )
        print("D405 world-to-camera extrinsic:")
        print(calibration.world_to_camera)
        print(f"D405 position in world (meters): {calibration.position_world}")
        print(
            f"Mean reprojection RMSE: {calibration.reprojection_error_px:.4f} px "
            f"from {calibration.sample_count} samples"
        )

        model, data, _ = build_calibrated_mujoco_robot(calibration)
        visualize_real_sim_overlay(
            camera,
            calibration,
            model,
            data,
            is_warp=args.is_warp,
        )
    except CalibrationAborted as exc:
        print(exc)
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
