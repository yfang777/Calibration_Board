"""Grouped RealSense camera worker process.

One spawned process owns every configured RealSense pipeline. Keeping all
pipelines in the same process avoids cross-process librealsense contention
while preserving process isolation from robot control and calibration logic.
"""

import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Sequence

import numpy as np

from .camera_appearance import apply_camera_appearance, load_camera_appearance


def _camera_group_main(
    pipe: Connection,
    rs_ids: list[str],
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    frame_timeout_ms: int,
    enable_depth: bool,
) -> None:
    # Import librealsense only in the spawned child. This gives the camera
    # owner a fresh runtime instead of inheriting parent-process USB state.
    import pyrealsense2 as rs

    appearance = load_camera_appearance()

    pipelines = {}
    aligners = {}
    depth_scales = {}
    active_rs_id: str | None = None

    try:
        intrinsics = {}
        for rs_id in rs_ids:
            active_rs_id = rs_id
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(rs_id)
            config.enable_stream(
                rs.stream.color,
                width,
                height,
                rs.format.bgr8,
                fps,
            )
            if enable_depth:
                config.enable_stream(
                    rs.stream.depth,
                    width,
                    height,
                    rs.format.z16,
                    fps,
                )

            profile = pipeline.start(config)
            pipelines[rs_id] = pipeline
            apply_camera_appearance(profile, appearance)
            if enable_depth:
                aligners[rs_id] = rs.align(rs.stream.color)
                depth_scales[rs_id] = float(
                    profile.get_device().first_depth_sensor().get_depth_scale()
                )

            color_stream = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            intrinsic = color_stream.get_intrinsics()
            intrinsics[rs_id] = np.array(
                [
                    [intrinsic.fx, 0.0, intrinsic.ppx],
                    [0.0, intrinsic.fy, intrinsic.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

        # Warm up only after all pipelines are active so startup exercises the
        # same configuration used for paired capture.
        for _ in range(warmup_frames):
            for rs_id in rs_ids:
                active_rs_id = rs_id
                pipelines[rs_id].wait_for_frames(frame_timeout_ms)

        pipe.send(("READY", intrinsics))
        while True:
            command = pipe.recv()
            if command == "STOP":
                break
            if command != "GET":
                raise RuntimeError(f"Unknown camera worker command: {command!r}")

            observations = {}
            for rs_id in rs_ids:
                active_rs_id = rs_id
                frames = pipelines[rs_id].wait_for_frames(frame_timeout_ms)
                if enable_depth:
                    frames = aligners[rs_id].process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError(f"No color frame from RealSense {rs_id}")
                color_bgr = np.asanyarray(color_frame.get_data()).copy()

                depth_m = None
                if enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        raise RuntimeError(f"No depth frame from RealSense {rs_id}")
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    depth_m = (
                        depth_raw.astype(np.float32) * depth_scales[rs_id]
                    )

                observations[rs_id] = {
                    "color": color_bgr,
                    "depth": depth_m,
                }

            pipe.send(("OBS", observations))
    except (BrokenPipeError, EOFError):
        pass
    except Exception as exc:
        try:
            pipe.send(("ERROR", active_rs_id, repr(exc)))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        for rs_id in reversed(rs_ids):
            pipeline = pipelines.get(rs_id)
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
        pipe.close()


class CameraGroupWorker:
    """Own multiple RealSense pipelines in one fresh child process."""

    def __init__(
        self,
        rs_ids: Sequence[str],
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 5,
        frame_timeout_ms: int = 15000,
        enable_depth: bool = True,
    ) -> None:
        context = mp.get_context("spawn")
        self.pipe, child_pipe = context.Pipe()
        self._child_pipe: Connection | None = child_pipe
        self._started = False
        self.process = context.Process(
            target=_camera_group_main,
            args=(
                child_pipe,
                [str(rs_id) for rs_id in rs_ids],
                width,
                height,
                fps,
                warmup_frames,
                frame_timeout_ms,
                enable_depth,
            ),
            name="CameraGroupWorker",
        )

    def start(self) -> None:
        try:
            self.process.start()
            self._started = True
        finally:
            if self._child_pipe is not None:
                self._child_pipe.close()
                self._child_pipe = None

    def join(self) -> None:
        if self._started:
            self.process.join()

    def is_alive(self) -> bool:
        return self._started and self.process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode

    def close(self) -> None:
        if self._child_pipe is not None:
            self._child_pipe.close()
            self._child_pipe = None
        self.pipe.close()
