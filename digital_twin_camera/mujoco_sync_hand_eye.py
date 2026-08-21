"""Write and test the hand-eye calibrated MuJoCo camera scene.

The MuJoCo world is the flipped ChArUco board world recorded in
``data/hand_eye/hand_eye_params.pkl``. ``write_scene_xml`` writes a calibrated
robot containing the link7-mounted D405 and a scene containing the fixed D455.
The source robot XML is never modified. ``test_mujoco_rgb`` reads the physical
robot and cameras and displays MuJoCo, intrinsic-warped RealSense, and overlay
images.
"""

from __future__ import annotations

import math
import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import cv2
import mujoco
import numpy as np
from xarm.wrapper import XArmAPI

from digital_twin_camera.cameras import RealSenseRgbCameras, warp_image


REPO = Path(__file__).resolve().parent.parent
HAND_EYE_DIR = REPO / "data/hand_eye"
HAND_EYE_PARAMS_PATH = HAND_EYE_DIR / "hand_eye_params.pkl"
SOURCE_ROBOT_XML = REPO / "assets/robot/xarm7/xarm7_with_wrist_direct.xml"
HAND_EYE_ROBOT_XML = (
    REPO / "assets/robot/xarm7/xarm7_with_wrist_hand_eye.xml"
)
SCENE_XML = REPO / "assets/mujoco_xml/scene_hand_eye.xml"
OVERLAY_PATH = HAND_EYE_DIR / "hand_eye_overlay.png"
ROBOT_IP = "192.168.1.196"

IMAGE_WIDTH = 848
IMAGE_HEIGHT = 480
IMAGE_FPS = 30
OVERLAY_WINDOW_SIZE = (1280, 480)
MUJOCO_CAMERA_RESOLUTION = (1280, 720)
MUJOCO_CAMERA_FOCAL_M = (1.93e-3, 1.93e-3)
MUJOCO_CAMERA_SENSOR_SIZE_M = (3896e-6, 2140e-6)
BOARD_SIZE = (12, 9)
BOARD_SQUARE_M = 0.03
CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])

FIXED_CAMERA_NAME = "outside_camera"
WRIST_CAMERA_NAME = "wrist_camera"
ROBOT_BASE_BODY_NAME = "link_base"
LINK7_BODY_NAME = "link7"


def _load_transform(value: object, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"{name} must have shape (4, 4), got {transform.shape}.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError(f"{name} has an invalid homogeneous row.")
    return transform


def load_hand_eye_params(
    params_path: Path = HAND_EYE_PARAMS_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Load the solved link7, robot-base, and fixed-camera transforms."""
    with params_path.open("rb") as params_file:
        params = pickle.load(params_file)
    if not params["quality_passed"]:
        raise RuntimeError(
            f"Hand-eye calibration did not pass quality checks: {params_path}"
        )
    return (
        _load_transform(
            params["link7_from_wrist_camera"],
            "link7_from_wrist_camera",
        ),
        _load_transform(params["base_from_world"], "base_from_world"),
        _load_transform(
            params["fixed_camera_from_world"],
            "fixed_camera_from_world",
        ),
        str(params["fixed_serial"]),
        str(params["wrist_serial"]),
    )


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.ascontiguousarray(rotation, dtype=np.float64).reshape(-1),
    )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _set_body_pose(element: ET.Element, parent_from_body: np.ndarray) -> None:
    element.set("pos", _format_vector(parent_from_body[:3, 3]))
    element.set(
        "quat",
        _format_vector(_rotation_to_quaternion(parent_from_body[:3, :3])),
    )


def _set_camera_pose(element: ET.Element, parent_from_cv_camera: np.ndarray) -> None:
    parent_from_mujoco_camera = parent_from_cv_camera.copy()
    parent_from_mujoco_camera[:3, :3] = (
        parent_from_cv_camera[:3, :3] @ CAMERA_AXIS_FLIP
    )
    _set_body_pose(element, parent_from_mujoco_camera)


def _set_camera_intrinsics(element: ET.Element) -> None:
    element.set("focal", _format_vector(MUJOCO_CAMERA_FOCAL_M))
    element.set("resolution", _format_vector(MUJOCO_CAMERA_RESOLUTION))
    element.set("sensorsize", _format_vector(MUJOCO_CAMERA_SENSOR_SIZE_M))


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in root.iter():
        if child in list(parent):
            return parent
    raise RuntimeError(f"Could not find the parent of XML element {child.tag!r}.")


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="unicode", xml_declaration=False)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n")


def _write_robot_xml(
    world_from_base: np.ndarray,
    link7_from_wrist_camera: np.ndarray,
    source_robot_xml: Path,
    output_robot_xml: Path,
) -> Path:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(source_robot_xml, parser=parser).getroot()
    root.set("model", "xarm7_with_wrist_hand_eye")
    base_body = root.find(f".//body[@name='{ROBOT_BASE_BODY_NAME}']")
    link7 = root.find(f".//body[@name='{LINK7_BODY_NAME}']")
    wrist_cameras = root.findall(f".//camera[@name='{WRIST_CAMERA_NAME}']")
    if base_body is None or link7 is None:
        raise RuntimeError("Source robot is missing link_base or link7.")
    if len(wrist_cameras) != 1:
        raise RuntimeError(
            f"Expected one {WRIST_CAMERA_NAME!r} in {source_robot_xml}; "
            f"found {len(wrist_cameras)}."
        )

    _set_body_pose(base_body, world_from_base)
    wrist_camera = wrist_cameras[0]
    _find_parent(root, wrist_camera).remove(wrist_camera)
    wrist_camera = ET.SubElement(
        link7,
        "camera",
        {"name": WRIST_CAMERA_NAME, "mode": "fixed"},
    )
    _set_camera_pose(wrist_camera, link7_from_wrist_camera)
    _set_camera_intrinsics(wrist_camera)
    _write_xml(root, output_robot_xml)
    return output_robot_xml


def _write_board_world_scene(
    world_from_fixed_camera: np.ndarray,
    robot_xml: Path,
    scene_xml: Path,
) -> Path:
    root = ET.Element("mujoco", {"model": "xarm7_hand_eye_board_world"})
    ET.SubElement(
        root,
        "include",
        {"file": os.path.relpath(robot_xml, scene_xml.parent)},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {"offwidth": str(IMAGE_WIDTH), "offheight": str(IMAGE_HEIGHT)},
    )
    worldbody = ET.SubElement(root, "worldbody")
    fixed_camera = ET.SubElement(
        worldbody,
        "camera",
        {"name": FIXED_CAMERA_NAME, "mode": "fixed"},
    )
    _set_camera_pose(fixed_camera, world_from_fixed_camera)
    _set_camera_intrinsics(fixed_camera)

    columns, rows = BOARD_SIZE
    for row in range(rows):
        for column in range(columns):
            ET.SubElement(
                worldbody,
                "site",
                {
                    "name": f"charuco_{column}_{row}",
                    "pos": _format_vector(
                        (column * BOARD_SQUARE_M, -row * BOARD_SQUARE_M, 0.0)
                    ),
                    "size": "0.003",
                    "rgba": "1 0 0 1",
                    "group": "2",
                },
            )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 -0.005",
            "size": "1 1 0.01",
            "rgba": "0.85 0.85 0.85 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _write_xml(root, scene_xml)
    return scene_xml


def write_scene_xml(
    link7_from_wrist_camera: np.ndarray,
    base_from_world: np.ndarray,
    fixed_camera_from_world: np.ndarray,
    *,
    source_robot_xml: Path = SOURCE_ROBOT_XML,
    output_robot_xml: Path = HAND_EYE_ROBOT_XML,
    scene_xml: Path = SCENE_XML,
) -> tuple[Path, Path]:
    """Spawn both calibrated cameras and write the board-world MuJoCo scene."""
    robot_xml = _write_robot_xml(
        _invert_transform(base_from_world),
        link7_from_wrist_camera,
        source_robot_xml,
        output_robot_xml,
    )
    scene_path = _write_board_world_scene(
        _invert_transform(fixed_camera_from_world),
        robot_xml,
        scene_xml,
    )
    return robot_xml, scene_path


def test_write_scene_xml(
    params_path: Path = HAND_EYE_PARAMS_PATH,
) -> tuple[Path, Path]:
    """Load the saved hand-eye result and generate the robot and scene XML."""
    (
        link7_from_wrist_camera,
        base_from_world,
        fixed_camera_from_world,
        fixed_serial,
        wrist_serial,
    ) = load_hand_eye_params(params_path)
    robot_xml, scene_xml = write_scene_xml(
        link7_from_wrist_camera,
        base_from_world,
        fixed_camera_from_world,
    )
    print(f"Fixed camera serial: {fixed_serial}")
    print(f"Wrist camera serial: {wrist_serial}")
    print(f"Wrote hand-eye robot: {robot_xml}")
    print(f"Wrote board-world scene: {scene_xml}")
    print("link7_from_wrist_camera:")
    print(link7_from_wrist_camera)
    return robot_xml, scene_xml


def _set_robot_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joints_deg: Sequence[float],
) -> None:
    for joint_index, angle_deg in enumerate(joints_deg, start=1):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"joint{joint_index}",
        )
        data.qpos[model.jnt_qposadr[joint_id]] = math.radians(float(angle_deg))
    mujoco.mj_forward(model, data)


def _read_robot_joints(arm: XArmAPI) -> list[float]:
    code, joints_deg = arm.get_servo_angle(is_radian=False)
    if code != 0:
        raise RuntimeError(f"xArm get_servo_angle failed with code {code}.")
    if len(joints_deg) != 7:
        raise RuntimeError(f"xArm returned {len(joints_deg)} joints instead of 7.")
    return [float(value) for value in joints_deg]


def _mujoco_intrinsic(
    model: mujoco.MjModel,
    camera_name: str,
) -> np.ndarray:
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    focal_x, focal_y = model.cam_intrinsic[camera_id, :2]
    sensor_width, sensor_height = model.cam_sensorsize[camera_id]
    return np.array(
        [
            [focal_x / sensor_width * IMAGE_WIDTH, 0.0, (IMAGE_WIDTH - 1) / 2.0],
            [0.0, focal_y / sensor_height * IMAGE_HEIGHT, (IMAGE_HEIGHT - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _render_bgr(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.putText(
        output,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _comparison_row(
    label: str,
    simulated: np.ndarray,
    real: np.ndarray,
) -> np.ndarray:
    overlay = cv2.addWeighted(simulated, 0.5, real, 0.5, 0.0)
    return np.hstack(
        (
            _label(simulated, f"{label}: MuJoCo"),
            _label(real, f"{label}: warped RealSense"),
            _label(overlay, f"{label}: overlay"),
        )
    )


def test_mujoco_rgb(
    *,
    show_gui: bool = True,
    max_frames: int = 0,
    params_path: Path = HAND_EYE_PARAMS_PATH,
    scene_xml: Path = SCENE_XML,
) -> None:
    """Overlay live D455/D405 RGB with the corresponding MuJoCo cameras."""
    if not show_gui and max_frames <= 0:
        raise ValueError("Headless RGB testing requires max_frames > 0.")

    _, _, _, fixed_serial, wrist_serial = load_hand_eye_params(params_path)
    camera_pairs = (
        (FIXED_CAMERA_NAME, fixed_serial),
        (WRIST_CAMERA_NAME, wrist_serial),
    )
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    real_cameras = RealSenseRgbCameras(
        [fixed_serial, wrist_serial],
        wrist_serial,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        fps=IMAGE_FPS,
    )
    arm = XArmAPI(ROBOT_IP, baud_checkset=False)
    renderers = {
        camera_name: mujoco.Renderer(
            model,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )
        for camera_name, _ in camera_pairs
    }
    mujoco_intrinsics = {
        camera_name: _mujoco_intrinsic(model, camera_name)
        for camera_name, _ in camera_pairs
    }
    window_name = "hand-eye RGB: MuJoCo | warped RealSense | overlay"

    try:
        if not arm.connected:
            raise RuntimeError(f"Could not connect to xArm at {ROBOT_IP}.")
        real_cameras.start()
        OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if show_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, *OVERLAY_WINDOW_SIZE)
        frame_index = 0
        while True:
            loop_started = time.perf_counter()
            observations = real_cameras.get_rgb()
            _set_robot_joints(model, data, _read_robot_joints(arm))
            rows = []
            for camera_name, serial in camera_pairs:
                simulated = _render_bgr(renderers[camera_name], data, camera_name)
                real = warp_image(
                    observations[serial],
                    real_cameras.intrinsics[serial],
                    mujoco_intrinsics[camera_name],
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                )
                rows.append(
                    _comparison_row(f"{camera_name} {serial}", simulated, real)
                )
            mosaic = np.vstack(rows)
            frame_index += 1

            if not show_gui:
                if frame_index == max_frames:
                    if not cv2.imwrite(str(OVERLAY_PATH), mosaic):
                        raise RuntimeError(f"Could not save overlay: {OVERLAY_PATH}")
                    break
                continue

            cv2.imshow(window_name, mosaic)
            elapsed = time.perf_counter() - loop_started
            delay_ms = max(1, int((1.0 / IMAGE_FPS - elapsed) * 1000.0))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("s"):
                if not cv2.imwrite(str(OVERLAY_PATH), mosaic):
                    raise RuntimeError(f"Could not save overlay: {OVERLAY_PATH}")
                print(f"Saved overlay: {OVERLAY_PATH}")
            if key in (27, ord("q")):
                break
            if max_frames > 0 and frame_index >= max_frames:
                break
    finally:
        for renderer in renderers.values():
            renderer.close()
        real_cameras.close()
        arm.disconnect()
        cv2.destroyAllWindows()

    if not show_gui:
        print(f"Saved overlay: {OVERLAY_PATH}")


if __name__ == "__main__":
    test_write_scene_xml()
    test_mujoco_rgb()
