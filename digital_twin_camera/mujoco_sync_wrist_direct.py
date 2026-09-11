"""Attach a saved wrist-camera calibration to link7 and test its MuJoCo view."""

from __future__ import annotations

import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import cv2
import mujoco
import numpy as np
import yaml

from digital_twin_camera import utils
from digital_twin_camera.cameras import (
    RealSenseRgbCameras,
    load_camera_serials,
    resize_like,
    warp_image,
)


REPO = Path(__file__).resolve().parent.parent
CAM_PARAMS_WRIST_PATH = REPO / "data/calib_params/cam_params_wrist.pkl"
MJ_XML_DIR = REPO / "assets/mujoco_xml/D405_direct"
SCENE_XML = MJ_XML_DIR / "scene_hand_direct.xml"
WRIST_ROBOT_XML = REPO / "assets/robot/xarm7/xarm7_with_wrist_direct.xml"
ROBOT_ENV_CONFIG = REPO / "robot_env_config.yaml"
TABLE_TEXTURE_PATH = (
    REPO.parent / "Codegen_Datagen/assets/materials/table_arrange/wood_e68650.png"
)
ROOM_LAYOUT = "table_wall"

# W = 640
# H = 480
W = 1280
H = 720
WRIST_CAMERA_NAME = "wrist_camera"
LINK7_BODY_NAME = "link7"
ROBOT_JOINTS_DEG = (0., -45.0, 0.0, 30, 0., 75, 0.0)
# ROBOT_JOINTS_DEG = (0., -75.0, 0.0, 10, 0., 60, 0.0)
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
CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
WRIST_CAMERA_FOCAL_M = (1.93e-3, 1.93e-3)
WRIST_CAMERA_RESOLUTION = (1280, 720)
WRIST_CAMERA_SENSOR_SIZE_M = (3896e-6, 2140e-6)
RGB_WINDOW = "Wrist camera: MuJoCo | D405 | overlay"


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        quaternion,
        np.ascontiguousarray(rotation, dtype=np.float64).reshape(-1),
    )
    return quaternion


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None


def load_room_layout(
    room_layout: str,
    config_path: Path = ROBOT_ENV_CONFIG,
) -> list[dict]:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)["room_layouts"][room_layout]


def _robot_tree_without_wrist_camera(
    robot_xml: Path,
) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(robot_xml, parser=parser)
    root = tree.getroot()
    cameras = root.findall(f".//camera[@name='{WRIST_CAMERA_NAME}']")
    if len(cameras) != 1:
        raise RuntimeError(
            f"Expected one {WRIST_CAMERA_NAME!r} in {robot_xml}; "
            f"found {len(cameras)}."
        )
    parent = _find_parent(root, cameras[0])
    if parent is None:
        raise RuntimeError(f"Could not find the parent of {WRIST_CAMERA_NAME!r}.")
    parent.remove(cameras[0])
    return tree


def _set_camera_attributes(
    camera: ET.Element,
    parent_from_wrist_camera: np.ndarray,
) -> None:
    """Set a MuJoCo camera from an OpenCV pose relative to its parent."""
    parent_from_mujoco_camera = parent_from_wrist_camera.copy()
    parent_from_mujoco_camera[:3, :3] = (
        parent_from_wrist_camera[:3, :3] @ CAMERA_AXIS_FLIP
    )
    camera.set("pos", _format_vector(parent_from_mujoco_camera[:3, 3]))
    camera.set(
        "quat",
        _format_vector(
            _rotation_to_quaternion(parent_from_mujoco_camera[:3, :3])
        ),
    )
    camera.set("focal", _format_vector(WRIST_CAMERA_FOCAL_M))
    camera.set("resolution", " ".join(map(str, WRIST_CAMERA_RESOLUTION)))
    camera.set("sensorsize", _format_vector(WRIST_CAMERA_SENSOR_SIZE_M))


def _new_scene_tree(
    robot_xml: Path,
    scene_xml: Path,
    room_layout: str,
) -> ET.ElementTree:
    root = ET.Element("mujoco", {"model": "wrist_camera_calibration"})
    ET.SubElement(
        root,
        "include",
        {"file": os.path.relpath(robot_xml, scene_xml.parent)},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {"offwidth": str(W), "offheight": str(H)},
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "default_table_wood",
            "type": "2d",
            "file": str(TABLE_TEXTURE_PATH),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "default_table_wood",
            "texture": "default_table_wood",
            "texrepeat": "1 1",
            "texuniform": "false",
            "specular": "0.1",
            "shininess": "0.15",
        },
    )
    worldbody = ET.SubElement(root, "worldbody")
    bodies = {}
    for geometry in load_room_layout(room_layout):
        parent = worldbody
        if "body" in geometry:
            body_name = geometry["body"]
            if body_name not in bodies:
                bodies[body_name] = ET.SubElement(
                    worldbody,
                    "body",
                    {"name": body_name},
                )
            parent = bodies[body_name]
        attributes = {
            key: " ".join(map(str, value)) if isinstance(value, list) else str(value)
            for key, value in geometry.items()
            if key != "body"
        }
        ET.SubElement(parent, "geom", attributes)
    columns, rows = BOARD_SIZE
    for row in range(rows):
        for column in range(columns):
            ET.SubElement(
                worldbody,
                "site",
                {
                    "name": f"charuco_{column}_{row}",
                    "pos": (
                        f"{column * BOARD_SQUARE_M:.4f} "
                        f"{-row * BOARD_SQUARE_M:.4f} 0"
                    ),
                    "size": "0.003",
                    "rgba": "1 0 0 1",
                    "group": "2",
                },
            )
    return ET.ElementTree(root)


def _close_gripper(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_name in GRIPPER_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint {joint_name!r} was not found.")
        data.qpos[model.jnt_qposadr[joint_id]] = GRIPPER_CLOSED_QPOS
    mujoco.mj_forward(model, data)


def _load_model(
    xml_path: Path,
    joints_deg: tuple[float, ...],
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if len(joints_deg) != 7:
        raise ValueError(f"Expected 7 robot joint angles, received {len(joints_deg)}.")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    utils.set_robot_pose(model, data, joints_deg)
    _close_gripper(model, data)
    return model, data


def _world_from_link7(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    link7_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        LINK7_BODY_NAME,
    )
    if link7_id < 0:
        raise RuntimeError(f"MuJoCo body {LINK7_BODY_NAME!r} was not found.")
    world_from_link7 = np.eye(4, dtype=np.float64)
    world_from_link7[:3, :3] = data.xmat[link7_id].reshape(3, 3)
    world_from_link7[:3, 3] = data.xpos[link7_id]
    return world_from_link7


def write_scene_xml(
    world_from_wrist_camera: np.ndarray,
    joints_deg: tuple[float, ...] = ROBOT_JOINTS_DEG,
    *,
    robot_xml: Path = WRIST_ROBOT_XML,
    scene_xml: Path = SCENE_XML,
    room_layout: str = ROOM_LAYOUT,
) -> tuple[Path, np.ndarray]:
    """Attach the calibrated camera to link7 and write the board-world scene."""
    robot_tree = _robot_tree_without_wrist_camera(robot_xml)
    model, data = _load_model(robot_xml, joints_deg)
    link7_from_wrist_camera = (
        np.linalg.inv(_world_from_link7(model, data)) @ world_from_wrist_camera
    )

    link7 = robot_tree.getroot().find(f".//body[@name='{LINK7_BODY_NAME}']")
    if link7 is None:
        raise RuntimeError(f"Body {LINK7_BODY_NAME!r} was not found in {robot_xml}.")
    camera = ET.SubElement(
        link7,
        "camera",
        {"name": WRIST_CAMERA_NAME, "mode": "fixed"},
    )
    _set_camera_attributes(camera, link7_from_wrist_camera)
    ET.indent(robot_tree, space="  ")
    robot_tree.write(robot_xml, encoding="utf-8", xml_declaration=False)

    scene_tree = _new_scene_tree(robot_xml, scene_xml, room_layout)
    scene_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(scene_tree, space="  ")
    scene_tree.write(scene_xml, encoding="utf-8", xml_declaration=False)
    return scene_xml, link7_from_wrist_camera


def load_wrist_camera_params(
    camera_params_path: Path = CAM_PARAMS_WRIST_PATH,
) -> tuple[str, np.ndarray]:
    """Load the configured D405 world-to-camera pose."""
    with camera_params_path.open("rb") as camera_params_file:
        camera_params = pickle.load(camera_params_file)

    _, wrist_serial = load_camera_serials()
    wrist_params = camera_params[wrist_serial]
    return wrist_serial, wrist_params["extrinsic"]


def _mujoco_camera_intrinsic(
    model: mujoco.MjModel,
    camera_id: int,
) -> np.ndarray:
    focal_x, focal_y = model.cam_intrinsic[camera_id, :2]
    principal_x, principal_y = model.cam_intrinsic[camera_id, 2:4]
    sensor_width, sensor_height = model.cam_sensorsize[camera_id]
    return np.array(
        [
            [
                focal_x / sensor_width * W,
                0.0,
                W / 2.0 + principal_x / sensor_width * W,
            ],
            [
                0.0,
                focal_y / sensor_height * H,
                H / 2.0 - principal_y / sensor_height * H,
            ],
            [0.0, 0.0, 1.0],
        ]
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        origin,
        font,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _window_was_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1.0
    except cv2.error:
        return True


def _compose_rgb_view(
    sim_bgr: np.ndarray,
    real_bgr: np.ndarray,
    serial: str,
    joints_deg: tuple[float, ...],
    is_warp: bool,
) -> np.ndarray:
    if sim_bgr.shape != real_bgr.shape:
        raise RuntimeError(
            "MuJoCo and D405 image shapes differ: "
            f"{sim_bgr.shape} versus {real_bgr.shape}."
        )
    sim_panel = sim_bgr.copy()
    real_panel = real_bgr.copy()
    overlay_panel = cv2.addWeighted(sim_bgr, 0.5, real_bgr, 0.5, 0.0)
    _put_text(sim_panel, "MuJoCo wrist_camera", (10, 25), 0.62)
    real_label = "Intrinsic-warped" if is_warp else "Raw"
    _put_text(real_panel, f"{real_label} D405 ({serial})", (10, 25), 0.62)
    _put_text(overlay_panel, "MuJoCo / D405 50/50 overlay", (10, 25), 0.62)

    panels = np.hstack((sim_panel, real_panel, overlay_panel))
    footer = np.zeros((38, panels.shape[1], 3), dtype=np.uint8)
    display = np.vstack((panels, footer))
    _put_text(
        display,
        f"Joints: [{_format_vector(joints_deg)}] | Esc exits",
        (10, H + 27),
        0.49,
    )
    return display


def test_mujoco_rgb(
    is_warp: bool = True,
    joints_deg: tuple[float, ...] = ROBOT_JOINTS_DEG,
    scene_xml: Path = SCENE_XML,
) -> None:
    """Overlay the current MuJoCo wrist-camera scene with the real D405."""
    model, data = _load_model(scene_xml, joints_deg)
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        WRIST_CAMERA_NAME,
    )
    if camera_id < 0:
        raise RuntimeError(f"MuJoCo camera {WRIST_CAMERA_NAME!r} was not found.")

    renderer = mujoco.Renderer(model, height=H, width=W)
    _, wrist_rs_id = load_camera_serials()
    real_camera = RealSenseRgbCameras(
        [wrist_rs_id],
        wrist_rs_id,
        image_size=(W, H),
    )
    try:
        real_camera.start()
        serial = real_camera.wrist_rs_id
        renderer.update_scene(data, camera=WRIST_CAMERA_NAME)
        sim_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        mujoco_intrinsic = _mujoco_camera_intrinsic(model, camera_id)

        cv2.namedWindow(RGB_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RGB_WINDOW, W * 3, H + 38)
        while True:
            real_bgr = real_camera.get_rgb()[serial]
            if is_warp:
                real_bgr = warp_image(
                    real_bgr,
                    real_camera.intrinsics[serial],
                    mujoco_intrinsic,
                    (W, H),
                )
            else:
                real_bgr = resize_like(real_bgr, sim_bgr)
            display = _compose_rgb_view(
                sim_bgr,
                real_bgr,
                serial,
                joints_deg,
                is_warp,
            )
            cv2.imshow(RGB_WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or _window_was_closed(RGB_WINDOW):
                break
    finally:
        real_camera.close()
        renderer.close()
        cv2.destroyAllWindows()


def test_write_scene_xml(
    joints_deg: tuple[float, ...] = ROBOT_JOINTS_DEG,
    camera_params_path: Path = CAM_PARAMS_WRIST_PATH,
) -> tuple[Path, np.ndarray]:
    """Load the saved calibration and write the link7-attached MuJoCo scene."""
    wrist_serial, world_to_wrist_camera = load_wrist_camera_params(
        camera_params_path
    )
    scene_xml, link7_from_wrist_camera = write_scene_xml(
        np.linalg.inv(world_to_wrist_camera),
        joints_deg,
    )

    print(f"Wrist camera serial: {wrist_serial}")
    print("world_to_wrist_camera:")
    print(world_to_wrist_camera)
    print("link7_from_wrist_camera:")
    print(link7_from_wrist_camera)
    print(f"Wrote robot XML: {WRIST_ROBOT_XML}")
    print(f"Wrote scene XML: {scene_xml}")
    return scene_xml, link7_from_wrist_camera


if __name__ == "__main__":
    test_write_scene_xml()
    test_mujoco_rgb(is_warp=True)
