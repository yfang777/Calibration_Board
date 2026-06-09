"""Overlay MuJoCo renders from the old camera XML vs the new camera XML
(regenerated from cam_params.pkl via digital_twin_camera.mujoco_sync).

Output: 2x3 grid -- rows = cameras, cols = Old | New | Overlay.
"""

import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco

from digital_twin_camera import mujoco_sync, utils
from digital_twin_camera.mujoco_sync import (
    MJ_XML_DIR,
    SCENE_XML,
    render_mj,
    write_camera_xml,
    write_scene_xml,
)

REPO = Path(__file__).resolve().parent.parent
CAM_PARAMS_PATH = REPO / "data/calib_params/cam_params.pkl"
OUT_DIR = REPO / "data/debug"
OUT_PATH = OUT_DIR / "overlay_mujoco_img.png"
OLD_CAMERA_FILENAME = "camera_old.xml"
NEW_CAMERA_FILENAME = "camera.xml"


def _render_all(camera_filename: str, cam_names: list[str], include_robot: bool):
    """Point scene.xml at the given camera include file, then render every cam_name."""
    mujoco_sync.CAMERA_XML = MJ_XML_DIR / camera_filename
    write_scene_xml(include_robot=include_robot)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    if include_robot:
        utils.set_robot_pose(model, data)
    else:
        mujoco.mj_forward(model, data)

    rgbs = {}
    for name in cam_names:
        rgb_bgr, _ = render_mj(model, data, name)
        rgbs[name] = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    return rgbs


def main(include_robot: bool = True):
    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)

    cam_names = [f"camera_{rs_id}" for rs_id in cam_params]

    # Old: use the checked-in camera_old.xml as-is.
    old_rgbs = _render_all(OLD_CAMERA_FILENAME, cam_names, include_robot)

    # New: regenerate camera.xml from cam_params.pkl, then render it.
    mujoco_sync.CAMERA_XML = MJ_XML_DIR / NEW_CAMERA_FILENAME
    write_camera_xml(cam_params)
    new_rgbs = _render_all(NEW_CAMERA_FILENAME, cam_names, include_robot)

    n = len(cam_names)
    _, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)
    for row, name in enumerate(cam_names):
        old = old_rgbs[name]
        new = new_rgbs[name]
        overlay = cv2.addWeighted(old, 0.5, new, 0.5, 0)

        axes[row, 0].imshow(old)
        axes[row, 0].set_title(f"old {name}")
        axes[row, 1].imshow(new)
        axes[row, 1].set_title(f"new {name}")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"overlay {name}")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
