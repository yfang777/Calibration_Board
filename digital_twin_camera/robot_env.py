"""xArm7 + RealSense environment for digital-twin camera alignment.

Connects to a real xArm7 over the network, spawns one ``CameraWorker`` per
RealSense device, and exposes ``move_to_joint`` + ``get_obs`` so callers can
drive the arm to a joint configuration and grab synchronized RGB+depth from
every camera.
"""

import multiprocessing as mp
import time

import numpy as np
from xarm.wrapper import XArmAPI

from .camera import CameraWorker


def angle2radian(angle):
    return np.deg2rad(np.array(angle, dtype=np.float32))


def radian2angle(radian):
    return np.rad2deg(np.array(radian, dtype=np.float32))


class Xarm7Env:
    def __init__(self, robot_ip="192.168.1.196", gripper_enable=True, connect_robot=True):
        self.workers: list[CameraWorker] = []
        self.pipes: list[mp.connection.Connection] = []
        self.rs_ids: list[str] = []

        self.init_servo_angle = np.array([0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0])
        self.gripper_enable = gripper_enable
        self.arm = None

        if connect_robot:
            self.arm = XArmAPI(robot_ip, baud_checkset=False)
            self.arm.clean_warn()
            self.arm.clean_error()
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)

            if self.gripper_enable:
                self.arm.set_gripper_enable(True)
                self.arm.set_gripper_mode(0)
                self.arm.clean_gripper_error()
                self.arm.set_gripper_position(850, wait=True)

    def set_cam(self, rs_ids, width=640, height=480, fps=30, settle_seconds=2.0, enable_depth=True):
        """Spawn one ``CameraWorker`` per serial in ``rs_ids``."""
        for rs_id in rs_ids:
            parent_conn, child_conn = mp.Pipe()
            worker = CameraWorker(
                child_conn,
                rs_id,
                width=width,
                height=height,
                fps=fps,
                enable_depth=enable_depth,
            )
            worker.start()
            self.workers.append(worker)
            self.pipes.append(parent_conn)
            self.rs_ids.append(rs_id)
        time.sleep(settle_seconds)

    def reset(self):
        if self.arm is None:
            return self.get_obs()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=self.init_servo_angle, isradian=False, wait=True)
        if self.gripper_enable:
            self.arm.set_gripper_position(850, wait=True)
        return self.get_obs()

    def move_to_joint(self, joints_deg, wait=True, settle_seconds=0.5):
        """Move the real arm to ``joints_deg`` (length-7, degrees)."""
        if self.arm is None:
            return self.get_obs()
        angles = np.array(joints_deg, dtype=np.float32)
        assert angles.shape == (7,), f"expected 7 joint angles, got {angles.shape}"
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=angles, isradian=False, wait=wait)
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        return self.get_obs()

    def get_obs(self):
        """Trigger every camera and collect ``{rs_id: {color, depth}}``."""
        for rs_id, pipe in zip(self.rs_ids, self.pipes):
            try:
                pipe.send("GET")
            except (BrokenPipeError, EOFError) as exc:
                raise RuntimeError(f"Camera worker for {rs_id} is not available") from exc

        obs = {}
        for pipe in self.pipes:
            msg = pipe.recv()
            if msg[0] == "ERROR":
                _, rs_id, err = msg
                raise RuntimeError(f"Camera worker for {rs_id} failed: {err}")
            rs_id, color_bgr, depth_m = msg
            obs[rs_id] = {"color": color_bgr, "depth": depth_m}
        return obs

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send("STOP")
            except (BrokenPipeError, EOFError):
                pass
        for worker in self.workers:
            worker.join()
        self.workers.clear()
        self.pipes.clear()
        self.rs_ids.clear()

        if self.arm is not None:
            if self.gripper_enable:
                self.arm.set_gripper_position(850, wait=True)
            self.arm.disconnect()
            self.arm = None
