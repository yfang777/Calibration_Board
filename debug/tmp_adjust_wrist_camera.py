"""Interactively align the MuJoCo wrist camera with the physical D405.

The window shows the MuJoCo render on the left, the raw RealSense image in the
middle, and their 50/50 overlay on the right. Camera translation and rotation
are applied in the camera's local frame. Press ``n`` to persist the current
pose to xarm7_with_wrist.xml.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np


# Allow both `python debug/tmp_adjust_wrist_camera.py` and
# `python -m debug.tmp_adjust_wrist_camera` from the repository root.
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
    load_camera_serials,
    write_scene_xml,
)


WINDOW_NAME = "Adjust wrist camera: MuJoCo | D405 | overlay"
DEFAULT_TRANSLATION_STEP_M = 0.001
DEFAULT_ROTATION_STEP_DEG = 0.5

_CAMERA_TAG_RE = re.compile(
    rf'<camera\b(?=[^>]*\bname\s*=\s*(["\']){re.escape(WRIST_CAMERA_NAME)}\1)[^>]*?/?>',
    flags=re.DOTALL,
)

# MuJoCo cameras look along local -Z, with local +X right and local +Y up.
_TRANSLATION_BY_KEY = {
    "w": np.array([0.0, 0.0, -1.0]),  # forward
    "s": np.array([0.0, 0.0, 1.0]),   # backward
    "a": np.array([-1.0, 0.0, 0.0]),  # left
    "d": np.array([1.0, 0.0, 0.0]),   # right
    "q": np.array([0.0, -1.0, 0.0]),  # down
    "e": np.array([0.0, 1.0, 0.0]),   # up
}

_ROTATION_BY_KEY = {
    "i": (np.array([1.0, 0.0, 0.0]), 1.0, "pitch up"),
    "k": (np.array([1.0, 0.0, 0.0]), -1.0, "pitch down"),
    "j": (np.array([0.0, 1.0, 0.0]), 1.0, "yaw left"),
    "l": (np.array([0.0, 1.0, 0.0]), -1.0, "yaw right"),
    "u": (np.array([0.0, 0.0, 1.0]), 1.0, "roll counterclockwise"),
    "o": (np.array([0.0, 0.0, 1.0]), -1.0, "roll clockwise"),
}


def _rotation_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def _camera_xyaxes(quat: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo camera quaternion back to its MJCF xyaxes values."""
    matrix = _rotation_matrix(quat)
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def _apply_translation(
    model: mujoco.MjModel,
    camera_id: int,
    local_direction: np.ndarray,
    step_m: float,
) -> None:
    local_delta = local_direction * step_m
    parent_delta = _rotation_matrix(model.cam_quat[camera_id]) @ local_delta
    model.cam_pos[camera_id] += parent_delta


def _apply_rotation(
    model: mujoco.MjModel,
    camera_id: int,
    local_axis: np.ndarray,
    angle_rad: float,
) -> None:
    delta_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(delta_quat, local_axis, angle_rad)

    # Post-multiplication applies the increment in the camera's local frame.
    updated_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(updated_quat, model.cam_quat[camera_id], delta_quat)
    model.cam_quat[camera_id] = updated_quat / np.linalg.norm(updated_quat)


def _format_values(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _replace_attribute(tag: str, attribute: str, value: str) -> str:
    pattern = re.compile(rf'(\b{re.escape(attribute)}\s*=\s*)(["\'])(.*?)\2', re.DOTALL)
    updated, count = pattern.subn(lambda match: f"{match.group(1)}{match.group(2)}{value}{match.group(2)}", tag)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {attribute!r} attribute in the {WRIST_CAMERA_NAME!r} tag; "
            f"found {count}."
        )
    return updated


def _validate_saved_pose(path: Path, expected_pos: np.ndarray, expected_quat: np.ndarray) -> None:
    candidate = mujoco.MjModel.from_xml_path(str(path))
    candidate_id = mujoco.mj_name2id(
        candidate, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_NAME
    )
    if candidate_id < 0:
        raise RuntimeError(f"Saved MJCF has no camera named {WRIST_CAMERA_NAME!r}.")
    if not np.allclose(candidate.cam_pos[candidate_id], expected_pos, atol=1e-10, rtol=0.0):
        raise RuntimeError("Saved wrist-camera position did not round-trip through MuJoCo.")
    if abs(float(np.dot(candidate.cam_quat[candidate_id], expected_quat))) < 1.0 - 1e-10:
        raise RuntimeError("Saved wrist-camera orientation did not round-trip through MuJoCo.")


def save_wrist_camera_pose(
    model: mujoco.MjModel,
    camera_id: int,
    path: Path = WRIST_ROBOT_XML,
) -> None:
    """Atomically update only wrist_camera's pos and xyaxes in the robot MJCF."""
    pos = model.cam_pos[camera_id].copy()
    quat = model.cam_quat[camera_id].copy()
    xyaxes = _camera_xyaxes(quat)

    source = path.read_text(encoding="utf-8")
    matches = list(_CAMERA_TAG_RE.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {WRIST_CAMERA_NAME!r} camera tag in {path}; found {len(matches)}."
        )

    match = matches[0]
    camera_tag = _replace_attribute(match.group(0), "pos", _format_values(pos))
    camera_tag = _replace_attribute(camera_tag, "xyaxes", _format_values(xyaxes))
    updated_source = source[: match.start()] + camera_tag + source[match.end() :]

    path_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".xml",
            delete=False,
        ) as temporary_file:
            temporary_file.write(updated_source)
            temporary_path = Path(temporary_file.name)

        os.chmod(temporary_path, path_mode)
        _validate_saved_pose(temporary_path, pos, quat)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Saved wrist camera pose to {path}")
    _print_pose("Saved", model, camera_id)


def _print_pose(prefix: str, model: mujoco.MjModel, camera_id: int) -> None:
    print(
        f"{prefix}: pos=[{_format_values(model.cam_pos[camera_id])}] "
        f"quat=[{_format_values(model.cam_quat[camera_id])}]"
    )


def _put_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _compose_display(
    sim_bgr: np.ndarray,
    real_bgr: np.ndarray,
    model: mujoco.MjModel,
    camera_id: int,
    translation_step_m: float,
    rotation_step_deg: float,
    rs_id: str,
) -> np.ndarray:
    if sim_bgr.shape != real_bgr.shape:
        raise RuntimeError(
            f"MuJoCo and D405 image shapes differ: {sim_bgr.shape} versus {real_bgr.shape}."
        )

    sim = sim_bgr.copy()
    real = real_bgr.copy()
    overlay = cv2.addWeighted(sim_bgr, 0.5, real_bgr, 0.5, 0.0)
    _put_text(sim, "MuJoCo wrist_camera", (10, 25), 0.65)
    _put_text(real, f"RealSense D405 ({rs_id})", (10, 25), 0.65)
    _put_text(overlay, "MuJoCo / D405 overlay", (10, 25), 0.65)

    images = np.hstack((sim, real, overlay))
    footer = np.zeros((92, images.shape[1], 3), dtype=np.uint8)
    display = np.vstack((images, footer))
    _put_text(
        display,
        "Move: W/S forward/back  A/D left/right  Q/E down/up | "
        "Rotate: I/K pitch  J/L yaw  U/O roll | N save | Esc exit",
        (10, H + 28),
        0.5,
    )
    pos = _format_values(model.cam_pos[camera_id])
    quat = _format_values(model.cam_quat[camera_id])
    _put_text(
        display,
        f"pos: [{pos}] m | step: {translation_step_m * 1000.0:g} mm, "
        f"{rotation_step_deg:g} deg",
        (10, H + 57),
        0.5,
    )
    _put_text(display, f"quat (wxyz): [{quat}]", (10, H + 82), 0.5)
    return display


def run(translation_step_m: float, rotation_step_deg: float) -> None:
    if translation_step_m <= 0.0:
        raise ValueError("Translation step must be positive.")
    if rotation_step_deg <= 0.0:
        raise ValueError("Rotation step must be positive.")

    _, wrist_rs_id = load_camera_serials()
    assert_realsense_connected([wrist_rs_id])

    # This utility always adjusts the D405-equipped model. Regenerating the
    # scene also ensures that it includes the latest saved wrist-camera pose.
    write_scene_xml(include_robot=True, robot_xml=WRIST_ROBOT_XML)
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    utils.set_robot_pose(model, data)

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA_NAME
    )
    if camera_id < 0:
        raise RuntimeError(
            f"MuJoCo camera {WRIST_CAMERA_NAME!r} was not found in {SCENE_XML}."
        )

    renderer = mujoco.Renderer(model, height=H, width=W)
    rs_camera = RealSenseRgbCameras([wrist_rs_id], wrist_rs_id=wrist_rs_id)
    rotation_step_rad = np.deg2rad(rotation_step_deg)

    print(f"Using D405 {wrist_rs_id}")
    print(f"Saving with N updates {WRIST_ROBOT_XML}")
    print("Controls: W/S forward/back, A/D left/right, Q/E down/up")
    print("          I/K pitch, J/L yaw, U/O roll, N save, Esc exit")
    _print_pose("Initial", model, camera_id)

    try:
        rs_camera.start()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, W * 3, H + 92)

        while True:
            renderer.update_scene(data, camera=WRIST_CAMERA_NAME)
            sim_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            real_bgr = rs_camera.get_rgb()[wrist_rs_id]
            display = _compose_display(
                sim_bgr,
                real_bgr,
                model,
                camera_id,
                translation_step_m,
                rotation_step_deg,
                wrist_rs_id,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 255:
                continue

            key_char = chr(key).lower()
            adjustment = None
            if key_char in _TRANSLATION_BY_KEY:
                _apply_translation(
                    model,
                    camera_id,
                    _TRANSLATION_BY_KEY[key_char],
                    translation_step_m,
                )
                adjustment = f"move {key_char.upper()}"
            elif key_char in _ROTATION_BY_KEY:
                axis, sign, description = _ROTATION_BY_KEY[key_char]
                _apply_rotation(model, camera_id, axis, sign * rotation_step_rad)
                adjustment = description
            elif key_char == "n":
                save_wrist_camera_pose(model, camera_id)

            if adjustment is not None:
                mujoco.mj_forward(model, data)
                _print_pose(adjustment, model, camera_id)
    finally:
        rs_camera.close()
        renderer.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively align MuJoCo's wrist_camera with the configured D405."
    )
    parser.add_argument(
        "--translation-step-mm",
        type=float,
        default=DEFAULT_TRANSLATION_STEP_M * 1000.0,
        help="Translation per key press in millimeters (default: %(default)s).",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=DEFAULT_ROTATION_STEP_DEG,
        help="Rotation per key press in degrees (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        translation_step_m=args.translation_step_mm / 1000.0,
        rotation_step_deg=args.rotation_step_deg,
    )


if __name__ == "__main__":
    main()
