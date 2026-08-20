"""RealSense capture helpers (cached pipelines) and robot-pose helpers."""

from pathlib import Path

import mujoco
import numpy as np
import pyrealsense2 as rs

W, H, FPS = 640, 480, 30

# --- robot ---------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
ROBOT_XML = REPO / "assets/robot/xarm7/xarm7_with_gripper.xml"

# Default xarm7 joint angles (degrees).
DEFAULT_JOINT_DEG = (0, -45, 0, 30, 0, 75, 0)

# Joint configurations swept by test_pointcloud_alignment (degrees).



def set_robot_pose(model, data, joints_deg=DEFAULT_JOINT_DEG):
    data.qpos[:len(joints_deg)] = np.deg2rad(joints_deg)
    mujoco.mj_forward(model, data)


# --- realsense -----------------------------------------------------------

_pipes: dict[str, rs.pipeline] = {}
_profiles: dict[str, rs.pipeline_profile] = {}
_depth_scale: dict[str, float] = {}
_align = rs.align(rs.stream.color)


def _ensure(rs_id: str) -> rs.pipeline:
    if rs_id in _pipes:
        return _pipes[rs_id]
    cfg = rs.config()
    cfg.enable_device(rs_id)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    _pipes[rs_id] = pipe
    _profiles[rs_id] = profile
    _depth_scale[rs_id] = float(profile.get_device().first_depth_sensor().get_depth_scale())
    return pipe


def _grab_aligned(rs_id: str):
    pipe = _ensure(rs_id)
    frames = _align.process(pipe.wait_for_frames(15000))
    return frames


def get_realsense_rgb(rs_id: str) -> np.ndarray:
    """Returns BGR uint8 image, HxWx3."""
    frames = _grab_aligned(rs_id)
    return np.asanyarray(frames.get_color_frame().get_data()).copy()


def get_realsense_depth(rs_id: str) -> np.ndarray:
    """Returns depth in meters, float32 HxW (aligned to color)."""
    frames = _grab_aligned(rs_id)
    depth_raw = np.asanyarray(frames.get_depth_frame().get_data())
    return depth_raw.astype(np.float32) * _depth_scale[rs_id]


def get_realsense_intrinsic(rs_id: str) -> np.ndarray:
    """3x3 K matrix of the color stream after alignment."""
    _ensure(rs_id)
    intr = _profiles[rs_id].get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def shutdown():
    for pipe in _pipes.values():
        pipe.stop()
    _pipes.clear()
    _profiles.clear()
    _depth_scale.clear()
