"""RealSense RGB camera discovery, configuration, and image helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config.yaml"
W, H = 640, 480
FPS = 30


def load_camera_serials() -> tuple[list[str], str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    fixed_rs_ids = [str(rs_id) for rs_id in config["rs_id"]["d455"].values()]
    wrist_rs_id = str(config["rs_id"]["d405"]["cam_1"])
    return fixed_rs_ids, wrist_rs_id


def connected_realsense_ids() -> set[str]:
    context = rs.context()
    return {
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    }


def assert_realsense_connected(rs_ids: list[str]) -> None:
    connected_rs_ids = connected_realsense_ids()
    for rs_id in rs_ids:
        if rs_id not in connected_rs_ids:
            raise RuntimeError(
                f"RealSense {rs_id} is not connected. "
                f"Connected serials: {sorted(connected_rs_ids)}"
            )


def warp_image(
    image: np.ndarray,
    source_intrinsic: np.ndarray,
    target_intrinsic: np.ndarray,
    size: tuple[int, int] = (W, H),
) -> np.ndarray:
    """Warp an image between two pinhole intrinsic coordinate frames."""
    source_to_target = (
        target_intrinsic @ np.linalg.inv(source_intrinsic)
    ).astype(np.float32)
    return cv2.warpPerspective(
        image,
        source_to_target,
        size,
        flags=cv2.INTER_LINEAR,
    )


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize an image to the reference image dimensions."""
    height, width = reference.shape[:2]
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def pinhole_remap(intrinsic: rs.intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Map undistorted output pixels to raw RealSense color pixels, keeping K.

    Invert the SDK's deprojection equations, including its inverse Brown
    model's tangential terms evaluated at radially scaled coordinates.
    See librealsense src/rs.cpp, rs2_deproject_pixel_to_point.
    """
    if intrinsic.model not in (
        rs.distortion.none,
        rs.distortion.brown_conrady,
        rs.distortion.inverse_brown_conrady,
    ):
        raise ValueError(f"Unsupported color distortion model: {intrinsic.model}")
    u, v = np.meshgrid(
        np.arange(intrinsic.width, dtype=np.float64),
        np.arange(intrinsic.height, dtype=np.float64),
    )
    x = (u - intrinsic.ppx) / intrinsic.fx
    y = (v - intrinsic.ppy) / intrinsic.fy
    if intrinsic.model != rs.distortion.none:
        k1, k2, p1, p2, k3 = intrinsic.coeffs
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xq, yq = x, y
        if intrinsic.model == rs.distortion.inverse_brown_conrady:
            xq, yq = x * radial, y * radial
        dx = 2 * p1 * xq * yq + p2 * (r2 + 2 * xq * xq)
        dy = 2 * p2 * xq * yq + p1 * (r2 + 2 * yq * yq)
        x, y = x * radial + dx, y * radial + dy
    return (
        (x * intrinsic.fx + intrinsic.ppx).astype(np.float32),
        (y * intrinsic.fy + intrinsic.ppy).astype(np.float32),
    )


class RealSenseRgbCameras:
    """RGB capture, optionally rectified to pinhole images with unchanged K."""

    def __init__(
        self,
        rs_ids: list[str],
        wrist_rs_id: str,
        frame_timeout_ms: int = 15000,
        image_size: tuple[int, int] = (W, H),
        fps: int = FPS,
        undistort: bool = False,
    ):
        self.rs_ids = list(rs_ids)
        self.wrist_rs_id = wrist_rs_id
        self.frame_timeout_ms = frame_timeout_ms
        self.width, self.height = image_size
        self.fps = fps
        self.undistort = undistort
        self.remap: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.pipes: dict[str, rs.pipeline] = {}
        self.intrinsics: dict[str, np.ndarray] = {}
        self.color_formats: dict[str, rs.format] = {}

    @classmethod
    def configured_wrist(cls) -> RealSenseRgbCameras:
        _, wrist_rs_id = load_camera_serials()
        return cls([wrist_rs_id], wrist_rs_id, image_size=(1280, 720))

    def start(self) -> RealSenseRgbCameras:
        assert_realsense_connected(self.rs_ids)

        try:
            for rs_id in self.rs_ids:
                color_format = (
                    rs.format.rgb8
                    if rs_id == self.wrist_rs_id
                    else rs.format.bgr8
                )
                pipe = rs.pipeline()
                config = rs.config()
                config.enable_device(rs_id)
                config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    color_format,
                    self.fps,
                )
                profile = pipe.start(config)
                self.pipes[rs_id] = pipe
                stream = profile.get_stream(
                    rs.stream.color
                ).as_video_stream_profile()
                intrinsic = stream.get_intrinsics()
                if self.undistort:
                    self.remap[rs_id] = pinhole_remap(intrinsic)
                self.intrinsics[rs_id] = np.array(
                    [
                        [intrinsic.fx, 0.0, intrinsic.ppx],
                        [0.0, intrinsic.fy, intrinsic.ppy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                self.color_formats[rs_id] = color_format

            for _ in range(5):
                self.get_rgb()
        except Exception:
            self.close()
            raise
        return self

    def get_rgb(self) -> dict[str, np.ndarray]:
        images = {}
        for rs_id in self.rs_ids:
            frames = self.pipes[rs_id].wait_for_frames(self.frame_timeout_ms)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"No color frame from RealSense {rs_id}")
            color = np.asanyarray(color_frame.get_data())
            if self.color_formats[rs_id] == rs.format.rgb8:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            if self.undistort:
                color = cv2.remap(color, *self.remap[rs_id], cv2.INTER_LINEAR)
            images[rs_id] = color.copy()
        return images

    def close(self) -> None:
        for pipe in self.pipes.values():
            pipe.stop()
        self.pipes.clear()
        self.intrinsics.clear()
        self.color_formats.clear()
        self.remap.clear()
