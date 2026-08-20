"""Compare MuJoCo digital-twin cameras against real RealSense cameras.

For each rs_id in cam_params.pkl:
  - convert (extrinsic, intrinsic) -> MJCF <camera>
  - render MuJoCo RGB + depth from that camera
  - capture realsense RGB + depth for the same rs_id
  - lift both depths to world-frame point clouds and overlay in PyVista
"""

import os
import pickle
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyrealsense2 as rs
import pyvista as pv
import yaml

from digital_twin_camera import utils
from digital_twin_camera.robot_env import Xarm7Env

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config.yaml"
CAM_PARAMS_PATH = REPO / "data/calib_params/cam_params.pkl"
MJ_XML_DIR = REPO / "assets/mujoco_xml"
CAMERA_XML = MJ_XML_DIR / "camera.xml"
SCENE_XML = MJ_XML_DIR / "scene.xml"
DEFAULT_ROBOT_XML = REPO / "assets/robot/xarm7/xarm7_with_gripper.xml"
WRIST_ROBOT_XML = REPO / "assets/robot/xarm7/xarm7_with_wrist.xml"

W, H = 640, 480
FPS = 30
WRIST_CAMERA_NAME = "wrist_camera"

# Charuco board parameters (must match calibrator.py)
BOARD_SIZE = (12, 9)   # (cols, rows) — number of squares
BOARD_SQUARE = 0.03    # square edge length in meters


def load_camera_serials() -> tuple[list[str], str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fixed_rs_ids = [str(rs_id) for rs_id in config["rs_id"]["d455"].values()]
    wrist_rs_id = str(config["rs_id"]["d405"]["cam_1"])
    return fixed_rs_ids, wrist_rs_id


def connected_realsense_ids() -> set[str]:
    context = rs.context()
    return {
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    }


def assert_realsense_connected(rs_ids: list[str]) -> None:
    connected_rs_ids = connected_realsense_ids()
    for rs_id in rs_ids:
        assert rs_id in connected_rs_ids, (
            f"RealSense {rs_id} is not connected. "
            f"Connected serials: {sorted(connected_rs_ids)}"
        )


def select_robot_xml(wrist_rs_id: str, connected_rs_ids: set[str] | None = None) -> tuple[Path, bool]:
    if connected_rs_ids is None:
        connected_rs_ids = connected_realsense_ids()
    use_wrist_robot = wrist_rs_id in connected_rs_ids
    robot_xml = WRIST_ROBOT_XML if use_wrist_robot else DEFAULT_ROBOT_XML
    return robot_xml, use_wrist_robot


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
           f'  focal="{fx_mj} {fy_mj}"\n'
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
    # if camera_name == WRIST_CAMERA_NAME:
    #     principal_x, principal_y = model.cam_intrinsic[camera_id][2:4]
    #     sensor_width, sensor_height = model.cam_sensorsize[camera_id]
    #     cx = width / 2 + principal_x / sensor_width * width
    #     cy = height / 2 - principal_y / sensor_height * height
    # else:
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


def write_scene_xml(include_robot: bool = True, robot_xml: Path = DEFAULT_ROBOT_XML) -> None:
    robot_rel = os.path.relpath(robot_xml, MJ_XML_DIR)
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


def render_mj_rgb(model, data, camera_name: str) -> np.ndarray:
    """Returns the named camera's RGB render as a BGR image."""
    renderer = mujoco.Renderer(model, height=H, width=W)
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render()  # RGB uint8
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        renderer.close()
    return rgb_bgr


def warp_image(rgb_rs: np.ndarray, K_rs: np.ndarray, K_mj: np.ndarray,
               size: tuple[int, int] = (W, H)) -> np.ndarray:
    """Warp a RealSense image into the MuJoCo camera's intrinsic frame via H = K_mj @ K_rs^{-1}."""
    H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
    return cv2.warpPerspective(rgb_rs, H_rs_to_mj, size, flags=cv2.INTER_LINEAR)


_apply_warp = warp_image  # alias so test_mujoco_rgb can use it even when shadowed by an arg name


def _resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize image to reference HxW if needed."""
    ref_h, ref_w = reference.shape[:2]
    if image.shape[:2] == (ref_h, ref_w):
        return image
    return cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)


class RealSenseRgbCameras:
    """Same-process RealSense RGB capture for the live RGB alignment view."""

    def __init__(self, rs_ids, wrist_rs_id, frame_timeout_ms=15000):
        self.rs_ids = list(rs_ids)
        self.wrist_rs_id = wrist_rs_id
        self.frame_timeout_ms = frame_timeout_ms
        self.pipes: dict[str, rs.pipeline] = {}
        self.intrinsics: dict[str, np.ndarray] = {}
        self.color_formats: dict[str, rs.format] = {}

    def start(self):
        assert_realsense_connected(self.rs_ids)

        try:
            for rs_id in self.rs_ids:
                if rs_id == self.wrist_rs_id:
                    color_format = rs.format.rgb8
                else:
                    color_format = rs.format.bgr8

                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(rs_id)
                cfg.enable_stream(rs.stream.color, W, H, color_format, FPS)
                profile = pipe.start(cfg)
                self.pipes[rs_id] = pipe
                stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = stream.get_intrinsics()
                intrinsic = np.array(
                    [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )

                self.intrinsics[rs_id] = intrinsic
                self.color_formats[rs_id] = color_format

            for _ in range(5):
                self.get_rgb()
        except Exception:
            self.close()
            raise
        return self

    def get_rgb(self) -> dict[str, np.ndarray]:
        images = {}
        for rs_id in self.rs_ids:
            frames = self.pipes[rs_id].wait_for_frames(self.frame_timeout_ms)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"No color frame from RealSense {rs_id}")
            color = np.asanyarray(color_frame.get_data())
            if self.color_formats[rs_id] == rs.format.rgb8:
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            images[rs_id] = color.copy()
        return images

    def close(self):
        for pipe in self.pipes.values():
            pipe.stop()
        self.pipes.clear()
        self.intrinsics.clear()
        self.color_formats.clear()


# --- main ----------------------------------------------------------------

def main():
    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)

    _, wrist_rs_id = load_camera_serials()
    robot_xml, _ = select_robot_xml(wrist_rs_id)
    write_camera_xml(cam_params)
    write_scene_xml(robot_xml=robot_xml)

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
    """Render MuJoCo RGB, capture RealSense RGB, and show an N x 3 grid:
       columns = sim | real | overlay, rows = scene cameras.

    If warp_image is True, warp the RealSense image into the MuJoCo intrinsic frame
    via H = K_mj @ K_rs^{-1} before overlaying.

    Fixed D455 cameras use names of the form camera_<realsense_serial>.
    The robot-mounted D405 uses the MuJoCo camera name wrist_camera and the
    D405 serial configured in config.yaml.
    """
    fixed_rs_ids, wrist_rs_id = load_camera_serials()
    robot_xml, use_wrist_robot = select_robot_xml(wrist_rs_id)
    write_scene_xml(include_robot=include_robot, robot_xml=robot_xml)
    print(f"Using robot model: {robot_xml.name}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    if include_robot and model.nq >= len(utils.DEFAULT_JOINT_DEG):
        utils.set_robot_pose(model, data)
    else:
        mujoco.mj_forward(model, data)

    cameras = [(f"camera_{rs_id}", rs_id) for rs_id in fixed_rs_ids]
    if include_robot and use_wrist_robot:
        cameras.append((WRIST_CAMERA_NAME, wrist_rs_id))

    for camera_name, _ in cameras:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        assert camera_id >= 0, f"MuJoCo camera {camera_name!r} was not found in {SCENE_XML}"

    sim_images = {}
    K_mj_by_rs = {}
    for camera_name, rs_id in cameras:
        sim_images[rs_id] = render_mj_rgb(model, data, camera_name)
        _, K_mj_by_rs[rs_id] = get_mj_cam(model, data, camera_name, W, H)

    rs_cameras = RealSenseRgbCameras(
        [rs_id for _, rs_id in cameras],
        wrist_rs_id=wrist_rs_id,
    )
    window_name = "mujoco rgb sync: sim | real | overlay"
    target_period = 1.0 / 30.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def label(image: np.ndarray, text: str) -> np.ndarray:
        out = image.copy()
        cv2.putText(out, text, (10, 24), font, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (10, 24), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    try:
        rs_cameras.start()
        K_real_by_rs = rs_cameras.intrinsics

        while True:
            loop_start = time.perf_counter()
            obs = rs_cameras.get_rgb()
            rows = []

            for camera_name, rs_id in cameras:
                sim = sim_images[rs_id]
                real = obs[rs_id]
                if warp_image:
                    real = _apply_warp(real, K_real_by_rs[rs_id], K_mj_by_rs[rs_id], (W, H))
                else:
                    real = _resize_like(real, sim)
                overlay = cv2.addWeighted(sim, 0.5, real, 0.5, 0.0)

                rows.append(
                    np.hstack([
                        label(sim, f"{camera_name} sim"),
                        label(real, f"{camera_name} real ({rs_id})"),
                        label(overlay, f"{camera_name} overlay"),
                    ])
                )

            mosaic = np.vstack(rows)
            cv2.imshow(window_name, mosaic)

            elapsed = time.perf_counter() - loop_start
            delay_ms = max(1, int((target_period - elapsed) * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        rs_cameras.close()
        cv2.destroyAllWindows()



def test_mujoco_pointcloud(warp_image: bool = True):
    """Sweep through SIM_POSES_DEG: drive the real arm + sim to each pose, lift
    sim and real depth into world-frame point clouds, and overlay them.

    Sim points use one base color per pose; real points use a contrasting color
    per pose so sim/real alignment is visible at a glance.

    If warp_image is True, warp the RealSense depth into the MuJoCo intrinsic
    frame (H = K_mj @ K_rs^{-1}) and lift it with K_mj; otherwise lift the raw
    RealSense depth with K_rs.

    Requires at least 2 cameras in cam_params.pkl; raises otherwise.
    """
    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)

    _, wrist_rs_id = load_camera_serials()
    robot_xml, _ = select_robot_xml(wrist_rs_id)
    write_scene_xml(robot_xml=robot_xml)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    utils.set_robot_pose(model, data)

    plotter = pv.Plotter()
    plotter.add_axes_at_origin(line_width=2)

    for rs_id, p in cam_params.items():
        # if rs_id == "244622300362":
        #     continue
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


if __name__ == "__main__":
    # main()
    test_mujoco_rgb()
