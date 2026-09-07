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


class RealSenseRgbCameras:
    """Same-process RGB capture for one or more configured RealSense cameras."""

    def __init__(
        self,
        rs_ids: list[str],
        wrist_rs_id: str,
        frame_timeout_ms: int = 15000,
        image_size: tuple[int, int] = (W, H),
        fps: int = FPS,
    ):
        self.rs_ids = list(rs_ids)
        self.wrist_rs_id = wrist_rs_id
        self.frame_timeout_ms = frame_timeout_ms
        self.width, self.height = image_size
        self.fps = fps
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
            images[rs_id] = color.copy()
        return images

    def close(self) -> None:
        for pipe in self.pipes.values():
            pipe.stop()
        self.pipes.clear()
        self.intrinsics.clear()
        self.color_formats.clear()
