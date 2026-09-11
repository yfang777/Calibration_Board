"""Generate a Datagen mug/rack scene and compare both live cameras.

Run from the repository root: python -m debug.tmp_dual_camera_overlay
The scene is static: set the physical robot to --joints and match object poses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import pyrealsense2 as rs

from digital_twin_camera.cameras import RealSenseRgbCameras, load_camera_serials, pinhole_remap, warp_image
from digital_twin_camera import mujoco_sync_wrist_direct as wrist

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "debug/dual_camera_overlay"
DATAGEN = REPO.parent / "Codegen_Datagen"
WINDOW = "D455 + D405: MuJoCo | Real | Overlay"


def generate_scene(output: Path, datagen: Path, joints: tuple, poses: dict) -> Path:
    """Write an isolated scene using current calibration and Datagen assets."""
    output.mkdir(parents=True, exist_ok=True)
    _, world_to_camera = wrist.load_wrist_camera_params()
    robot_tree = wrist._robot_tree_without_wrist_camera(wrist.WRIST_ROBOT_XML)
    for element in robot_tree.getroot().iter():
        if "file" in element.attrib:
            element.set("file", str((wrist.WRIST_ROBOT_XML.parent / element.attrib["file"]).resolve()))
    model, data = wrist._load_model(wrist.WRIST_ROBOT_XML, joints)
    parent_from_camera = np.linalg.inv(wrist._world_from_link7(model, data)) @ np.linalg.inv(world_to_camera)
    link7 = robot_tree.getroot().find(".//body[@name='link7']")
    camera = ET.SubElement(link7, "camera", name="wrist_camera", mode="fixed")
    wrist._set_camera_attributes(camera, parent_from_camera)
    robot_xml = output / "robot.xml"
    ET.indent(robot_tree)
    robot_tree.write(robot_xml)
    scene_xml = output / "scene.xml"
    tree = wrist._new_scene_tree(robot_xml, scene_xml, wrist.ROOM_LAYOUT)
    root = tree.getroot()
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true",
                  diffuse="0.4 0.4 0.4", castshadow="false")
    # Preserve the existing fixed camera's intrinsic convention and saved pose.
    fixed_ids, _ = load_camera_serials()
    with (REPO / "data/calib_params/cam_params.pkl").open("rb") as file:
        fixed_params = pickle.load(file)
    for serial in fixed_ids:
        params = fixed_params[serial]
        camera_world = np.linalg.inv(params["extrinsic"])
        rotation = camera_world[:3, :3] @ wrist.CAMERA_AXIS_FLIP
        intrinsic = params["intrinsic"]
        ET.SubElement(world, "camera", name=f"camera_{serial}", mode="fixed",
                      pos=wrist._format_vector(camera_world[:3, 3]),
                      quat=wrist._format_vector(wrist._rotation_to_quaternion(rotation)),
                      resolution="640 480", sensorsize="1 1",
                      focal=f"{intrinsic[0, 0] / 640} {intrinsic[1, 1] / 480}")
    for name, category in [("mug", "039_mug"), ("rack", "040_rack")]:
        asset_dir = datagen / "assets/objects/hang_mug" / category / "1"
        object_tree = ET.parse(asset_dir / f"{category}_1_decomposed_template.xml")
        for element in object_tree.getroot().iter():
            if "file" in element.attrib:
                element.set("file", str((asset_dir / element.attrib["file"]).resolve()))
        body = object_tree.getroot().find("worldbody/body")
        x, y, z, yaw = poses[name]
        angle = np.deg2rad(yaw) / 2
        body.set("pos", wrist._format_vector([x, y, z]))
        body.set("quat", wrist._format_vector([np.cos(angle), 0, 0, np.sin(angle)]))
        object_xml = output / f"{name}.xml"
        ET.indent(object_tree)
        object_tree.write(object_xml)
        ET.SubElement(root, "include", file=object_xml.name)
    ET.indent(tree)
    tree.write(scene_xml)
    return scene_xml


def label(image: np.ndarray, text: str) -> np.ndarray:
    panel = image.copy()
    cv2.putText(panel, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(panel, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_images(output: Path, images: dict, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        path = output / f"{name}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write {path}")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved comparison: {output}")


def run(args: argparse.Namespace) -> None:
    poses = {"mug": args.mug, "rack": args.rack}
    scene = generate_scene(args.output, args.datagen, tuple(args.joints), poses)
    print(f"Generated {scene}\nBoard-world object poses [x, y, z, yaw degrees]: {poses}")
    print(f"Physical robot must be stationary at joints (degrees): {args.joints}")
    model, data = wrist._load_model(scene, tuple(args.joints))
    fixed_ids, wrist_id = load_camera_serials()
    names = {serial: f"camera_{serial}" for serial in fixed_ids}
    names[wrist_id] = "wrist_camera"
    renderer = mujoco.Renderer(model, height=wrist.H, width=wrist.W)
    sims, target_intrinsics = {}, {}
    try:
        for serial, name in names.items():
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            target_intrinsics[serial] = wrist._mujoco_camera_intrinsic(model, camera_id)
            renderer.update_scene(data, camera=name)
            sims[serial] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    finally:
        renderer.close()
    metadata = {"scene": str(scene), "joints_deg": args.joints, "object_poses_xyz_yaw": poses,
                "wrist_calibration": str(wrist.CAM_PARAMS_WRIST_PATH),
                "fixed_calibration": str(REPO / "data/calib_params/cam_params.pkl"),
                "target_intrinsics": {s: k.tolist() for s, k in target_intrinsics.items()}}
    if args.render_only:
        save_images(args.output / "renders", {f"{s}_sim": im for s, im in sims.items()}, metadata)
        return

    # Both pipelines stay active. Frames are read consecutively; no hardware sync.
    fixed = RealSenseRgbCameras(fixed_ids, wrist_id, image_size=(640, 480))
    hand = RealSenseRgbCameras([wrist_id], wrist_id, image_size=(wrist.W, wrist.H))
    corrected = True
    try:
        fixed.start()
        hand.start()
        profile = hand.pipes[wrist_id].get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
        maps = pinhole_remap(profile.get_intrinsics())
        source_intrinsics = {**fixed.intrinsics, **hand.intrinsics}
        metadata["source_intrinsics"] = {s: k.tolist() for s, k in source_intrinsics.items()}
        metadata["d405_distortion_model"] = str(profile.get_intrinsics().model)
        metadata["d405_distortion_coefficients"] = list(profile.get_intrinsics().coeffs)
        if not args.snapshot:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1600, 650)
            cv2.createTrackbar("Simulation %", WINDOW, 50, 100, lambda value: None)
        print("U: toggle D405 undistortion | S: save both comparisons | Q/Esc: exit")
        while True:
            raw = {**fixed.get_rgb(), **hand.get_rgb()}
            undistorted = cv2.remap(raw[wrist_id], *maps, cv2.INTER_LINEAR)
            alpha = 0.5 if args.snapshot else cv2.getTrackbarPos("Simulation %", WINDOW) / 100
            rows, images = [], {}
            for serial in names:
                sim = sims[serial]
                warped_raw = warp_image(raw[serial], source_intrinsics[serial], target_intrinsics[serial], (wrist.W, wrist.H))
                if serial == wrist_id:
                    warped_corrected = warp_image(undistorted, source_intrinsics[serial], target_intrinsics[serial], (wrist.W, wrist.H))
                    real = warped_corrected if corrected else warped_raw
                    mode = "undistorted + warped" if corrected else "raw + warped"
                    images[f"{serial}_corrected_overlay"] = cv2.addWeighted(sim, alpha, warped_corrected, 1-alpha, 0)
                    images[f"{serial}_uncorrected_overlay"] = cv2.addWeighted(sim, alpha, warped_raw, 1-alpha, 0)
                    images[f"{serial}_undistorted"] = undistorted
                else:
                    real, mode = warped_raw, "existing D455 warp"
                overlay = cv2.addWeighted(sim, alpha, real, 1-alpha, 0)
                images.update({f"{serial}_raw": raw[serial], f"{serial}_sim": sim,
                               f"{serial}_warped": real, f"{serial}_overlay": overlay})
                row = np.hstack([label(sim, f"{serial}: MuJoCo"), label(real, mode), label(overlay, "Overlay")])
                rows.append(cv2.resize(row, (1920, 360)))
            display = np.vstack(rows)
            images["comparison"] = display
            metadata["d405_undistortion_enabled"] = corrected
            metadata["simulation_alpha"] = alpha
            if args.snapshot:
                save_images(args.output / "capture", images, metadata)
                break
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("u"):
                corrected = not corrected
            if key == ord("s"):
                save_images(args.output / "capture", images, metadata)
    finally:
        hand.close()
        fixed.close()
        if not args.snapshot:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datagen", type=Path, default=DATAGEN)
    parser.add_argument("--joints", type=float, nargs=7, default=list(wrist.ROBOT_JOINTS_DEG))
    parser.add_argument("--mug", type=float, nargs=4, default=[0.23, -0.30, 0, 180], metavar=("X", "Y", "Z", "YAW"))
    parser.add_argument("--rack", type=float, nargs=4, default=[0.32, 0, 0, -80], metavar=("X", "Y", "Z", "YAW"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-only", action="store_true", help="Generate scene and render both cameras without hardware or GUI.")
    mode.add_argument("--snapshot", action="store_true", help="Save one live dual-camera comparison without a GUI.")
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
