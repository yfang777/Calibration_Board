"""Calibrate the wrist D405 in the calibration-board world frame."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from digital_twin_camera.cameras import RealSenseRgbCameras


REPO = Path(__file__).resolve().parent
CAM_PARAMS_WRIST_PATH = REPO / "data/calib_params/cam_params_wrist.pkl"

BOARD_SIZE = (12, 9)
BOARD_SQUARE_M = 0.03
BOARD_MARKER_M = 0.022
BOARD_FROM_WORLD = np.diag([1.0, -1.0, -1.0])

CALIBRATION_WINDOW = "D405 ChArUco wrist calibration"


@dataclass(frozen=True)
class WristCameraCalibration:
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


def calibrate_wrist_camera(
    camera: RealSenseRgbCameras,
    *,
    sample_count: int = 10,
    min_corners: int = 35,
    max_reprojection_error_px: float = 0.2,
    visualize: bool = True,
) -> WristCameraCalibration:
    """Estimate the wrist D405 pose in the calibration-board world frame.

    ``camera`` must already be started.  The board-world convention matches
    ``calibrator.py``: world X equals native board X, while world Y and Z are
    flipped from the ChArUco board frame.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    if min_corners < 4:
        raise ValueError("min_corners must be at least 4.")
    if max_reprojection_error_px <= 0.0:
        raise ValueError("max_reprojection_error_px must be positive.")
    serial = camera.wrist_rs_id
    if serial not in camera.intrinsics:
        raise RuntimeError(f"RealSense {serial} is not started.")

    intrinsic = camera.intrinsics[serial].copy()
    distortion = np.zeros(5, dtype=np.float64)
    board, detector = _make_charuco_detector()
    chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)

    rotations_world_to_camera: list[np.ndarray] = []
    translations_world_to_camera: list[np.ndarray] = []
    accepted_errors: list[float] = []
    accepted_corner_counts: list[int] = []

    if visualize:
        cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CALIBRATION_WINDOW, camera.width, camera.height)

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
                    board_to_camera = cv2.Rodrigues(raw_rvec)[0]
                    rotations_world_to_camera.append(
                        board_to_camera @ BOARD_FROM_WORLD
                    )
                    translations_world_to_camera.append(raw_tvec.reshape(3).copy())
                    accepted_errors.append(error_px)
                    accepted_corner_counts.append(detected_count)

            if visualize:
                annotated = color_bgr.copy()
                if marker_corners is not None and len(marker_corners) > 0:
                    cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
                if charuco_corners is not None and detected_count > 0:
                    cv2.aruco.drawDetectedCornersCharuco(
                        annotated,
                        charuco_corners,
                        charuco_ids,
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
                    (10, camera.height - 15),
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
    return WristCameraCalibration(
        serial=serial,
        intrinsic=intrinsic,
        distortion=distortion,
        world_to_camera=world_to_camera,
        reprojection_error_px=float(np.mean(accepted_errors)),
        detected_corners=int(round(float(np.mean(accepted_corner_counts)))),
        sample_count=sample_count,
    )


def save_wrist_camera_params(
    calibration: WristCameraCalibration,
    output_path: Path = CAM_PARAMS_WRIST_PATH,
) -> Path:
    """Save the wrist calibration using the repository camera-parameter schema."""
    camera_params = {
        calibration.serial: {
            "extrinsic": calibration.world_to_camera,
            "intrinsic": calibration.intrinsic,
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(camera_params, output_file)
    return output_path


def main() -> None:
    camera = RealSenseRgbCameras.configured_wrist()
    try:
        camera.start()
        calibration = calibrate_wrist_camera(camera)
    finally:
        camera.close()

    output_path = save_wrist_camera_params(calibration)
    print(f"Wrist camera serial: {calibration.serial}")
    print("World-to-wrist-camera extrinsic:")
    print(calibration.world_to_camera)
    print("Wrist camera intrinsic:")
    print(calibration.intrinsic)
    print(f"Wrist camera position in world (meters): {calibration.position_world}")
    print(
        f"Mean reprojection RMSE: {calibration.reprojection_error_px:.4f} px "
        f"from {calibration.sample_count} samples"
    )
    print(f"Saved wrist camera parameters: {output_path}")


if __name__ == "__main__":
    main()
