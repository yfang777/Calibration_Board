"""xArm7 + grouped RealSense environment for camera alignment.

Connects to a real xArm7, runs every RealSense pipeline in one dedicated
camera process, and exposes robot movement plus paired RGB/depth capture.
"""

import time

import numpy as np
from xarm.wrapper import XArmAPI

from .camera import CameraGroupWorker


def angle2radian(angle):
    return np.deg2rad(np.array(angle, dtype=np.float32))


def radian2angle(radian):
    return np.rad2deg(np.array(radian, dtype=np.float32))


class Xarm7Env:
    def __init__(self, robot_ip="192.168.1.196", gripper_enable=True, connect_robot=True):
        self.camera_worker: CameraGroupWorker | None = None
        self.rs_ids: list[str] = []
        self.camera_intrinsics: dict[str, np.ndarray] = {}

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

    @staticmethod
    def _raise_camera_error(message) -> None:
        _, rs_id, error = message
        camera_label = rs_id if rs_id is not None else "unknown"
        raise RuntimeError(f"Camera worker for {camera_label} failed: {error}")

    def set_cam(
        self,
        rs_ids,
        width=640,
        height=480,
        fps=30,
        settle_seconds=2.0,
        enable_depth=True,
    ):
        """Start one worker that owns every RealSense in ``rs_ids``."""
        if self.camera_worker is not None:
            raise RuntimeError("RealSense cameras have already been started.")

        serials = [str(rs_id) for rs_id in rs_ids]
        if not serials:
            raise ValueError("At least one RealSense serial is required.")
        if len(set(serials)) != len(serials):
            raise ValueError(f"RealSense serials must be unique: {serials}")
        if settle_seconds < 0:
            raise ValueError("settle_seconds cannot be negative.")

        worker = CameraGroupWorker(
            serials,
            width=width,
            height=height,
            fps=fps,
            enable_depth=enable_depth,
        )
        self.camera_worker = worker
        self.rs_ids = serials
        try:
            worker.start()
            message = worker.pipe.recv()
            if message[0] == "ERROR":
                self._raise_camera_error(message)
            if message[0] != "READY":
                raise RuntimeError(
                    f"Unexpected camera worker startup response: {message[0]!r}"
                )
            self.camera_intrinsics = {
                rs_id: np.asarray(intrinsic, dtype=np.float64)
                for rs_id, intrinsic in message[1].items()
            }
            if set(self.camera_intrinsics) != set(self.rs_ids):
                raise RuntimeError(
                    "Camera worker did not return intrinsics for every RealSense."
                )
            if settle_seconds > 0:
                time.sleep(settle_seconds)
        except (BrokenPipeError, EOFError) as exc:
            self._close_camera_worker()
            raise RuntimeError("Camera worker exited during startup.") from exc
        except BaseException:
            self._close_camera_worker()
            raise

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
        """Trigger one grouped worker and collect ``{rs_id: {color, depth}}``."""
        worker = self.camera_worker
        if worker is None:
            raise RuntimeError("RealSense cameras have not been started.")
        try:
            worker.pipe.send("GET")
            message = worker.pipe.recv()
        except (BrokenPipeError, EOFError) as exc:
            raise RuntimeError("Camera worker is not available.") from exc

        if message[0] == "ERROR":
            self._raise_camera_error(message)
        if message[0] != "OBS":
            raise RuntimeError(
                f"Unexpected camera worker capture response: {message[0]!r}"
            )
        observations = message[1]
        if set(observations) != set(self.rs_ids):
            raise RuntimeError(
                "Grouped camera observation did not contain every RealSense."
            )
        return observations

    def _close_camera_worker(self) -> None:
        worker = self.camera_worker
        if worker is None:
            return
        try:
            if worker.is_alive():
                try:
                    worker.pipe.send("STOP")
                except (BrokenPipeError, EOFError):
                    pass
            worker.join()
        finally:
            worker.close()
            self.camera_worker = None
            self.rs_ids.clear()
            self.camera_intrinsics.clear()

    def close(self):
        self._close_camera_worker()

        if self.arm is not None:
            if self.gripper_enable:
                self.arm.set_gripper_position(850, wait=True)
            self.arm.disconnect()
            self.arm = None
