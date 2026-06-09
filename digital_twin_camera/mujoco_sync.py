"""Compare MuJoCo digital-twin cameras against real RealSense cameras.

For each rs_id in cam_params.pkl:
  - convert (extrinsic, intrinsic) -> MJCF <camera>
  - render MuJoCo RGB + depth from that camera
  - capture realsense RGB + depth for the same rs_id
  - lift both depths to world-frame point clouds and overlay in PyVista
"""

import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pyvista as pv

from digital_twin_camera import utils
from digital_twin_camera.robot_env import Xarm7Env

REPO = Path(__file__).resolve().parent.parent
CAM_PARAMS_PATH = REPO / "data/calib_params/cam_params.pkl"
MJ_XML_DIR = REPO / "assets/mujoco_xml"
CAMERA_XML = MJ_XML_DIR / "camera.xml"
SCENE_XML = MJ_XML_DIR / "scene.xml"

W, H = 640, 480

# Charuco board parameters (must match calibrator.py)
BOARD_SIZE = (12, 9)   # (cols, rows) — number of squares
BOARD_SQUARE = 0.03    # square edge length in meters


# --- camera conversion ---------------------------------------------------

def convert_to_mj_camera(extrinsic: np.ndarray, intrinsic: np.ndarray,
                         rs_id: str, resolution=(W, H)) -> str:
    # Decompose extrinsic
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    # If your extrinsic is camera_T_world, use this conversion:

    R_c2b = R.T
    t_c2b = -R.T @ t
    R_b2w = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    t_b2w = np.array([0, 0, 0])
    T_c2b = np.eye(4)
    T_c2b[:3, :3] = R_c2b
    T_c2b[:3, 3] = t_c2b
    T_b2w = np.eye(4)
    T_b2w[:3, :3] = R_b2w
    T_b2w[:3, 3] = t_b2w
    # Compose
    T_c2w = T_b2w @ T_c2b
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]
    # Camera axes in world
    x_axis = (R_c2w[:, 0] / np.linalg.norm(R_c2w[:, 0])).astype(float)
    y_axis = -(R_c2w[:, 1] / np.linalg.norm(R_c2w[:, 1])).astype(float)

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    w, h = resolution
    sx, sy = 1, 1

    # Convert from pixel units -> MuJoCo units
    fx_mj = fx / (w * sx)
    fy_mj = fy / (h * sy)
    cx_mj = cx / (w * sx)
    cy_mj = cy / (h * sy)

    xml = (f'<camera name="camera_{rs_id}"\n'
           f'  pos="{float(t_c2w[0]):.8f} {float(t_c2w[1]):.8f} {float(t_c2w[2]):.8f}"\n'
           f'  mode="fixed"\n'
           f'  resolution="{w} {h}" sensorsize="{sx} {sy}"\n'
           f'  focal="{fx_mj} {fy_mj}" \n'
           f'  xyaxes="{x_axis[0]:.8f} {x_axis[1]:.8f} {x_axis[2]:.8f}  '
           f'{y_axis[0]:.8f} {y_axis[1]:.8f} {y_axis[2]:.8f}"/>')

    return xml

def get_mj_cam(model, data, camera_name, width, height):
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = data.cam_xmat[camera_id].reshape(3, 3).T
    R = np.eye(3)
    R[0, :] = weird_R[0, :]
    R[1, :] = -weird_R[1, :]
    R[2, :] = -weird_R[2, :]
    cam_pos = data.cam_xpos[camera_id]
    t = -np.dot(R, cam_pos)
    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R
    ex_mat[:3, 3] = t

    fx = (model.cam_intrinsic[camera_id][0] / model.cam_sensorsize[camera_id][0]
          * model.cam_resolution[camera_id][0])
    fy = (model.cam_intrinsic[camera_id][1] / model.cam_sensorsize[camera_id][1]
          * model.cam_resolution[camera_id][1])
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    in_mat = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]])
    return ex_mat, in_mat



def write_camera_xml(cam_params: dict) -> None:
    MJ_XML_DIR.mkdir(parents=True, exist_ok=True)
    tags = [
        convert_to_mj_camera(p["extrinsic"], p["intrinsic"], rs_id)
        for rs_id, p in cam_params.items()
    ]
    body = "\n".join("    " + line for tag in tags for line in tag.splitlines())
    CAMERA_XML.write_text(
        "<mujoco>\n"
        "  <worldbody>\n"
        f"{body}\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


def write_scene_xml(include_robot: bool = True) -> None:
    robot_rel = os.path.relpath(utils.ROBOT_XML, MJ_XML_DIR)
    cam_rel = CAMERA_XML.name

    cols, rows = BOARD_SIZE
    sq = BOARD_SQUARE
    sites = []
    # OpenCV CharucoBoard inner corners: (i*sq, j*sq, 0) in board frame,
    # i in [1..cols-1], j in [1..rows-1]. Calibration world = Rx180 @ board,
    # so world coords = (i*sq, -j*sq, 0).
    for j in range(0, rows):
        for i in range(0, cols):
            x = i * sq
            y = -j * sq
            sites.append(
                f'    <site name="charuco_{i}_{j}" pos="{x:.4f} {y:.4f} 0" '
                f'size="0.003" rgba="1 0 0 1" group="2"/>'
            )
    sites_str = "\n".join(sites)

    robot_include = f'  <include file="{robot_rel}"/>\n' if include_robot else ""

    SCENE_XML.write_text(
        '<mujoco model="digital_twin">\n'
        f'{robot_include}'
        f'  <include file="{cam_rel}"/>\n'
        '  <worldbody>\n'
        '    <geom name="floor" type="plane" pos="0 0 -0.005" size="1 1 0.01" '
        'rgba="0.85 0.85 0.85 1"/>\n'
        f'{sites_str}\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    )


# --- pointcloud lifting --------------------------------------------------

def depth_to_world_points(depth: np.ndarray, rgb_bgr: np.ndarray, K: np.ndarray,
                          extrinsic_w2c: np.ndarray,
                          max_depth: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Lift a depth map (meters) into a world-frame Nx3 point cloud + Nx3 RGB (0..1).

    Pixels with depth <= 0 or depth >= max_depth are dropped.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    valid = (z > 0) & (z < max_depth)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1)

    R_w2c = extrinsic_w2c[:3, :3]
    t_w2c = extrinsic_w2c[:3, 3]
    pts_world = (R_w2c.T @ (pts_cam - t_w2c).T).T

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[valid].astype(np.float32) / 255.0
    return pts_world, rgb


# --- mujoco rendering ----------------------------------------------------

def render_mj(model, data, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_bgr, depth_m) for the named camera."""
    rgb_renderer = mujoco.Renderer(model, height=H, width=W)
    rgb_renderer.update_scene(data, camera=camera_name)
    rgb = rgb_renderer.render()  # RGB uint8
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    depth_renderer = mujoco.Renderer(model, height=H, width=W)
    depth_renderer.enable_depth_rendering()
    depth_renderer.update_scene(data, camera=camera_name)
    depth = depth_renderer.render().astype(np.float32)
    return rgb_bgr, depth


def warp_image(rgb_rs: np.ndarray, K_rs: np.ndarray, K_mj: np.ndarray,
               size: tuple[int, int] = (W, H)) -> np.ndarray:
    """Warp a RealSense image into the MuJoCo camera's intrinsic frame via H = K_mj @ K_rs^{-1}."""
    H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
    return cv2.warpPerspective(rgb_rs, H_rs_to_mj, size, flags=cv2.INTER_LINEAR)


_apply_warp = warp_image  # alias so test_mujoco_rgb can use it even when shadowed by an arg name


# --- main ----------------------------------------------------------------

def main():
    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)

    write_camera_xml(cam_params)
    write_scene_xml()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    utils.set_robot_pose(model, data)

    plotter = pv.Plotter()
    plotter.add_axes_at_origin(line_width=2)

    for rs_id, p in cam_params.items():
        print("rs_id", rs_id)
        extrinsic = p["extrinsic"]
        cam_name = f"camera_{rs_id}"

        mj_rgb, mj_depth = render_mj(model, data, cam_name)
        mj_extrinsic, K_mj = get_mj_cam(model, data, cam_name, W, H)
        rs_rgb = utils.get_realsense_rgb(rs_id)
        rs_depth = utils.get_realsense_depth(rs_id)
        K_real = utils.get_realsense_intrinsic(rs_id)

        mj_pts, _ = depth_to_world_points(mj_depth, mj_rgb, K_mj, mj_extrinsic)
        rs_pts, rs_col = depth_to_world_points(rs_depth, rs_rgb, K_real, extrinsic)

        plotter.add_mesh(
            pv.PolyData(mj_pts),
            color=(0.2, 0.6, 1.0),
            point_size=2,
            label=f"sim {rs_id}",
        )
        rs_cloud = pv.PolyData(rs_pts)
        rs_cloud["RGB"] = (rs_col * 255).astype(np.uint8)
        plotter.add_mesh(
            rs_cloud,
            scalars="RGB",
            rgb=True,
            point_size=2,
            label=f"real {rs_id}",
        )

        print(f"[{rs_id}] sim pts={len(mj_pts)}  real pts={len(rs_pts)}")

    utils.shutdown()
    plotter.add_legend()
    plotter.show()


# --- tests ---------------------------------------------------------------

def test_mujoco_rgb(include_robot: bool = True, warp_image: bool = True):
    """Render MuJoCo RGB, capture RealSense RGB, and show a 3xN grid:
       row 0 = sim, row 1 = real, row 2 = overlay.

    If warp_image is True, warp the RealSense image into the MuJoCo intrinsic frame
    via H = K_mj @ K_rs^{-1} before overlaying.

    Requires at least 2 cameras in cam_params.pkl; raises otherwise.
    """
    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)


    write_camera_xml(cam_params)
    write_scene_xml(include_robot=include_robot)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    if include_robot:
        utils.set_robot_pose(model, data)
    else:
        mujoco.mj_forward(model, data)

    n = len(cam_params)
    _, axes = plt.subplots(3, n, figsize=(5 * n, 12), squeeze=False)

    for col, (rs_id, _) in enumerate(cam_params.items()):
        cam_name = f"camera_{rs_id}"
        mj_rgb_bgr, _ = render_mj(model, data, cam_name)
        rs_rgb_bgr = utils.get_realsense_rgb(rs_id)

        if warp_image:
            _, K_mj = get_mj_cam(model, data, cam_name, W, H)
            K_rs = utils.get_realsense_intrinsic(rs_id)
            rs_rgb_bgr = _apply_warp(rs_rgb_bgr, K_rs, K_mj)

        mj_rgb = cv2.cvtColor(mj_rgb_bgr, cv2.COLOR_BGR2RGB)
        rs_rgb = cv2.cvtColor(rs_rgb_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(mj_rgb, 0.5, rs_rgb, 0.5, 0)

        axes[0, col].imshow(mj_rgb)
        axes[0, col].set_title(f"sim {cam_name}")
        axes[1, col].imshow(rs_rgb)
        axes[1, col].set_title(f"real {rs_id}")
        axes[2, col].imshow(overlay)
        axes[2, col].set_title(f"overlay {rs_id}")
        for row in range(3):
            axes[row, col].axis("off")

        print(f"[{rs_id}] sim={mj_rgb.shape} real={rs_rgb.shape}")

    utils.shutdown()
    plt.tight_layout()
    plt.show()


def test_pointcloud_alignment(warp_image: bool = True):
    """Sweep through SIM_POSES_DEG: drive the real arm + sim to each pose, lift
    sim and real depth into world-frame point clouds, and overlay them.

    Sim points use one base color per pose; real points use a contrasting color
    per pose so sim/real alignment is visible at a glance.

    If warp_image is True, warp the RealSense depth into the MuJoCo intrinsic
    frame (H = K_mj @ K_rs^{-1}) and lift it with K_mj; otherwise lift the raw
    RealSense depth with K_rs.

    Requires at least 2 cameras in cam_params.pkl; raises otherwise.
    """

    SIM_POSES_DEG = [
        (0.0,  -9.6,   0.3,  37.6, -0.6, 47.2,   0.2),
        (-9.9,  1.3, -10.1,  39.0, -0.6, 37.4, -20.4),
        (-9.3, 19.6,  -6.3,  64.0,  2.6, 44.2, -18.0),
    ]

    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)

    if len(cam_params) < 2:
        raise ValueError(
            f"test_pointcloud_alignment needs >= 2 cameras, found {len(cam_params)} in {CAM_PARAMS_PATH}"
        )

    write_camera_xml(cam_params)
    write_scene_xml()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    env = Xarm7Env()
    env.set_cam(list(cam_params.keys()))

    # sim = cool tone, real = warm tone (same colors across poses; one window per pose).
    SIM_COLOR = (0.2, 0.6, 1.0)
    REAL_COLOR = (1.0, 0.4, 0.3)

    try:
        for p_idx, pose_deg in enumerate(SIM_POSES_DEG):
            utils.set_robot_pose(model, data, pose_deg)
            real_obs = env.move_to_joint(pose_deg)

            plotter = pv.Plotter(title=f"pose {p_idx}: {pose_deg}")
            plotter.add_axes_at_origin(line_width=2)

            for rs_id, p in cam_params.items():
                cam_name = f"camera_{rs_id}"

                mj_rgb, mj_depth = render_mj(model, data, cam_name)
                mj_extrinsic, K_mj = get_mj_cam(model, data, cam_name, W, H)
                mj_pts, _ = depth_to_world_points(mj_depth, mj_rgb, K_mj, mj_extrinsic)

                rs_rgb = real_obs[rs_id]["color"]
                rs_depth = real_obs[rs_id]["depth"]
                K_rs = p["intrinsic"]
                if warp_image:
                    H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
                    rs_rgb = cv2.warpPerspective(rs_rgb, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)
                    rs_depth = cv2.warpPerspective(rs_depth, H_rs_to_mj, (W, H), flags=cv2.INTER_NEAREST)
                    K_for_lift = K_mj
                else:
                    K_for_lift = K_rs
                rs_pts, _ = depth_to_world_points(
                    rs_depth, rs_rgb, K_for_lift, p["extrinsic"]
                )

                plotter.add_mesh(
                    pv.PolyData(mj_pts),
                    color=SIM_COLOR,
                    point_size=2,
                    label=f"sim {rs_id}",
                )
                plotter.add_mesh(
                    pv.PolyData(rs_pts),
                    color=REAL_COLOR,
                    point_size=2,
                    label=f"real {rs_id}",
                )
                print(
                    f"[pose{p_idx} {rs_id}] sim pts={len(mj_pts)}  real pts={len(rs_pts)}"
                )

            plotter.add_legend()
            print(f"[pose{p_idx}] close the window to advance to the next pose")
            plotter.show()
    finally:
        env.close()


if __name__ == "__main__":
    test_mujoco_rgb()
