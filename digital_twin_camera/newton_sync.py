"""Compare Newton digital-twin cameras against real RealSense cameras.

For each rs_id in cam_params.pkl:
  - build a Newton scene (table + optional robot + charuco-corner sites)
  - convert (extrinsic, intrinsic) -> SensorTiledCamera pose + FOV
  - render the tiled camera's RGB (+ optional ray-length depth)
  - capture realsense RGB / depth for the same rs_id
  - either show sim/real/overlay (3xN grid) or lift both depths to world-frame
    point clouds and overlay them in PyVista.
"""

import math
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import warp as wp

import newton
import newton.solvers
from newton.sensors import SensorTiledCamera

from digital_twin_camera import utils
from digital_twin_camera.robot_env import Xarm7Env

REPO = Path(__file__).resolve().parent.parent
CAM_PARAMS_PATH = REPO / "data/calib_params/cam_params.pkl"
URDF_PATH = REPO / "assets/robot/xarm7/xarm7_with_gripper.urdf"

W, H = 640, 480

# Charuco board parameters (must match calibrator.py)
BOARD_SIZE = (12, 9)   # (cols, rows) — number of squares
BOARD_SQUARE = 0.03    # square edge length in meters
SITE_RADIUS = 0.003

# Table: thin box, top surface at z=0 (matches the floor plane the MuJoCo path used).
TABLE_SIZE = (1.0, 1.0, 0.01)
TABLE_POS = (0.0, 0.0, -0.005)

NUM_ARM_JOINTS = 7


# --- camera helpers ------------------------------------------------------

def _fov_deg_from_intrinsic(intrinsic: np.ndarray, height: int = H) -> float:
    """Vertical FOV (deg) from fy and image height — matches the TiledCamera YAML form."""
    fy = float(intrinsic[1, 1])
    return math.degrees(2.0 * math.atan(0.5 * height / fy))


def _intrinsic_from_fov_deg(fov_deg: float, width: int = W, height: int = H) -> np.ndarray:
    """K matrix matching SensorTiledCamera's pinhole rays (square pixels, principal point centered)."""
    fy = (0.5 * height) / math.tan(0.5 * math.radians(fov_deg))
    fx = fy
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _world_from_cam_gl(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV cam_from_world extrinsic -> (R_c2w_gl, cam_pos_world).

    Newton's camera is OpenGL (+X right, +Y up, -Z fwd); OpenCV is (+X right,
    +Y down, +Z fwd), so we flip Y/Z columns of R_c2w.
    """
    T_w2c = np.asarray(extrinsic, dtype=np.float64)
    R_c2w = T_w2c[:3, :3].T
    cam_pos = -R_c2w @ T_w2c[:3, 3]
    R_c2w_gl = R_c2w * np.array([1.0, -1.0, -1.0])
    return R_c2w_gl, cam_pos


def _camera_transform(extrinsic: np.ndarray) -> wp.array:
    """OpenCV cam_from_world extrinsic -> (1,1) warp array of OpenGL camera transforms."""
    R_c2w_gl, cam_pos = _world_from_cam_gl(extrinsic)
    xf = wp.transformf(
        wp.vec3f(*cam_pos.tolist()),
        wp.quat_from_matrix(wp.mat33f(R_c2w_gl)),
    )
    return wp.array([[xf]], dtype=wp.transformf)


# --- pointcloud lifting --------------------------------------------------

def newton_depth_to_world_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    rays_local: np.ndarray,
    R_c2w_gl: np.ndarray,
    cam_pos: np.ndarray,
    max_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a Newton depth map (ray-length, meters) to world-frame points + RGB (0..1).

    Newton's tiled camera stores ``closest_hit.distance`` at each pixel, so the
    camera-frame point is simply ``ray_dir * distance``; world frame follows by
    applying R_c2w_gl and cam_pos.

    rays_local: (H, W, 3) unit ray directions in camera-local OpenGL frame.
    rgb:        (H, W, 3) uint8 RGB (already in RGB order).

    Pixels with depth <= 0 or depth >= max_depth are dropped.
    """
    valid = (depth > 0) & (depth < max_depth)
    pts_local = rays_local[valid] * depth[valid, None]
    pts_world = pts_local @ R_c2w_gl.T + cam_pos
    rgb_valid = rgb[valid].astype(np.float32) / 255.0
    return pts_world, rgb_valid


def depth_to_world_points(
    depth: np.ndarray,
    rgb_bgr: np.ndarray,
    K: np.ndarray,
    extrinsic_w2c: np.ndarray,
    max_depth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a perpendicular-Z depth map (meters) to world-frame points + RGB (0..1).

    Used for RealSense depth, which is true pinhole Z-depth (not ray length).
    extrinsic_w2c is the OpenCV cam_from_world extrinsic.

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


# --- scene ---------------------------------------------------------------

class NewtonCalibScene:
    """Newton scene with table + optional xarm7 + charuco sites + per-camera tiled sensors."""

    def __init__(
        self,
        cam_params: dict,
        include_robot: bool = True,
        joints_deg=utils.DEFAULT_JOINT_DEG,
    ):
        self.cam_params = cam_params
        self.include_robot = include_robot

        self._site_shape_ids: list[int] = []
        self._tiled_cams: dict[str, SensorTiledCamera] = {}
        self._tiled_cam_rays: dict[str, wp.array] = {}
        self._tiled_cam_color: dict[str, object] = {}
        self._tiled_cam_depth: dict[str, object] = {}
        self._fov_deg: dict[str, float] = {}
        self._rays_local_np: dict[str, np.ndarray] = {}
        self._R_c2w_gl: dict[str, np.ndarray] = {}
        self._cam_pos: dict[str, np.ndarray] = {}

        self._build_model(joints_deg)
        self._init_cameras()

    def _build_model(self, joints_deg) -> None:
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)

        # Table
        tw, td, th = TABLE_SIZE
        table_mesh = newton.Mesh.create_box(
            tw * 0.5, td * 0.5, th * 0.5,
            duplicate_vertices=True, compute_normals=False,
            compute_uvs=False, compute_inertia=True,
        )
        table_mesh._color = (0.8, 0.8, 0.8)
        scene.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(*TABLE_POS), wp.quat_identity()),
            mesh=table_mesh,
            label="table",
        )

        # Robot
        if self.include_robot:
            robot_builder = newton.ModelBuilder()
            robot_builder.add_urdf(
                str(URDF_PATH),
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                enable_self_collisions=False,
                parse_visuals_as_colliders=False,
            )
            arm_q = np.deg2rad(joints_deg).tolist()
            for i, q in enumerate(arm_q[:NUM_ARM_JOINTS]):
                if i < len(robot_builder.joint_q):
                    robot_builder.joint_q[i] = q
            scene.add_builder(robot_builder)

        # Charuco corner sites — one small sphere per inner-corner location.
        # OpenCV CharucoBoard inner corners: (i*sq, j*sq, 0) in board frame.
        # Calibration world = Rx180 @ board, so world coords = (i*sq, -j*sq, 0).
        cols, rows = BOARD_SIZE
        sq = BOARD_SQUARE
        site_mesh = newton.Mesh.create_sphere(
            radius=SITE_RADIUS,
            compute_normals=False, compute_uvs=False, compute_inertia=False,
        )
        for j in range(rows):
            for i in range(cols):
                sid = scene.add_shape_mesh(
                    body=-1,
                    xform=wp.transform(
                        wp.vec3(i * sq, -j * sq, 0.0), wp.quat_identity()
                    ),
                    mesh=site_mesh,
                    label=f"charuco_{i}_{j}",
                )
                self._site_shape_ids.append(sid)

        scene.color()
        self.model = scene.finalize(requires_grad=False)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _init_cameras(self) -> None:
        site_rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for name, p in self.cam_params.items():
            fov_deg = _fov_deg_from_intrinsic(p["intrinsic"])
            self._fov_deg[name] = fov_deg

            sensor = SensorTiledCamera(
                model=self.model,
                options=SensorTiledCamera.Options(
                    default_light=True,
                    default_light_shadows=False,
                    checkerboard_texture=True,
                    backface_culling=False,
                ),
            )
            self._tiled_cams[name] = sensor
            rays = sensor.compute_pinhole_camera_rays(
                width=W, height=H, camera_fovs=np.deg2rad(fov_deg),
            )
            self._tiled_cam_rays[name] = rays
            # rays shape: (camera_count, H, W, 2); index 1 = unit direction in camera-local frame.
            self._rays_local_np[name] = rays.numpy()[0, :, :, 1]

            self._tiled_cam_color[name] = sensor.create_color_image_output(
                width=W, height=H, camera_count=1,
            )
            self._tiled_cam_depth[name] = sensor.create_depth_image_output(
                width=W, height=H, camera_count=1,
            )

            R_c2w_gl, cam_pos = _world_from_cam_gl(p["extrinsic"])
            self._R_c2w_gl[name] = R_c2w_gl
            self._cam_pos[name] = cam_pos

            shape_colors_np = sensor.render_context.shape_colors.numpy()
            for sid in self._site_shape_ids:
                shape_colors_np[sid] = site_rgba
            sensor.render_context.shape_colors.assign(shape_colors_np)

    def set_joints(self, joints_deg) -> None:
        """Update arm joint positions and re-evaluate FK so the next render uses the new pose."""
        if not self.include_robot:
            return
        arm_q_rad = np.deg2rad(np.asarray(joints_deg, dtype=np.float32))
        joint_q_np = self.model.joint_q.numpy().copy()
        n = min(len(arm_q_rad), NUM_ARM_JOINTS, len(joint_q_np))
        joint_q_np[:n] = arm_q_rad[:n]
        self.model.joint_q.assign(joint_q_np)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _update_sensor(self, name: str, with_depth: bool) -> None:
        sensor = self._tiled_cams[name]
        cam_tf = _camera_transform(self.cam_params[name]["extrinsic"])
        sensor.update(
            self.state,
            cam_tf,
            self._tiled_cam_rays[name],
            color_image=self._tiled_cam_color[name],
            depth_image=self._tiled_cam_depth[name] if with_depth else None,
        )

    def render(self, name: str) -> np.ndarray:
        """Render the named camera and return an (H, W, 3) uint8 RGB array."""
        self._update_sensor(name, with_depth=False)
        rgba_buf = wp.array(shape=(H, W, 4), dtype=wp.uint8)
        self._tiled_cams[name].flatten_color_image_to_rgba(
            self._tiled_cam_color[name], rgba_buf, 1,
        )
        return rgba_buf.numpy()[:, :, :3]

    def render_rgbd(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Render the named camera and return (rgb HxWx3 uint8, depth HxW float32 ray-length meters)."""
        self._update_sensor(name, with_depth=True)
        rgba_buf = wp.array(shape=(H, W, 4), dtype=wp.uint8)
        self._tiled_cams[name].flatten_color_image_to_rgba(
            self._tiled_cam_color[name], rgba_buf, 1,
        )
        rgb = rgba_buf.numpy()[:, :, :3]
        depth = self._tiled_cam_depth[name].numpy()[0, 0].astype(np.float32)
        return rgb, depth

    def lift_to_world(
        self, name: str, depth: np.ndarray, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lift a Newton-rendered (rgb, depth) pair into a world-frame point cloud + RGB."""
        return newton_depth_to_world_points(
            depth, rgb,
            self._rays_local_np[name], self._R_c2w_gl[name], self._cam_pos[name],
        )

    def get_intrinsic(self, name: str) -> np.ndarray:
        return _intrinsic_from_fov_deg(self._fov_deg[name])


# --- tests ---------------------------------------------------------------

def test_newton_rgb(include_robot: bool = True, warp_image: bool = True):
    """Render Newton RGB, capture RealSense RGB, and show a 3xN grid:
       row 0 = sim, row 1 = real, row 2 = overlay.

    If warp_image is True, warp the RealSense image into the Newton camera's
    intrinsic frame via H = K_newton @ K_rs^{-1} before overlaying.
    """
    with open(CAM_PARAMS_PATH, "rb") as f:
        cam_params = pickle.load(f)

    scene = NewtonCalibScene(cam_params, include_robot=include_robot)

    n = len(cam_params)
    _, axes = plt.subplots(3, n, figsize=(5 * n, 12), squeeze=False)

    for col, rs_id in enumerate(cam_params.keys()):
        sim_rgb = scene.render(rs_id)
        rs_bgr = utils.get_realsense_rgb(rs_id)

        if warp_image:
            K_newton = scene.get_intrinsic(rs_id)
            K_rs = utils.get_realsense_intrinsic(rs_id)
            H_rs_to_newton = (K_newton @ np.linalg.inv(K_rs)).astype(np.float32)
            rs_bgr = cv2.warpPerspective(
                rs_bgr, H_rs_to_newton, (W, H), flags=cv2.INTER_LINEAR
            )

        rs_rgb = cv2.cvtColor(rs_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(sim_rgb, 0.5, rs_rgb, 0.5, 0)

        axes[0, col].imshow(sim_rgb)
        axes[0, col].set_title(f"sim camera_{rs_id}")
        axes[1, col].imshow(rs_rgb)
        axes[1, col].set_title(f"real {rs_id}")
        axes[2, col].imshow(overlay)
        axes[2, col].set_title(f"overlay {rs_id}")
        for row in range(3):
            axes[row, col].axis("off")

        print(f"[{rs_id}] sim={sim_rgb.shape} real={rs_rgb.shape}")

    utils.shutdown()
    plt.tight_layout()
    plt.show()


def test_pointcloud_alignment():
    """Sweep through SIM_POSES_DEG: drive the real arm + sim to each pose, lift
    sim and real depth into world-frame point clouds, and overlay them.

    Sim points use one base color per pose; real points use a contrasting color
    so sim/real alignment is visible at a glance.

    Sim depth uses Newton's native ray-length depth + the per-pixel ray
    directions returned by ``compute_pinhole_camera_rays`` (no K-warp needed).
    Real depth is RealSense perpendicular-Z depth, lifted with the per-camera
    intrinsic from cam_params.
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

    scene = NewtonCalibScene(cam_params, include_robot=True)

    env = Xarm7Env()
    env.set_cam(list(cam_params.keys()))

    SIM_COLOR = (0.2, 0.6, 1.0)
    REAL_COLOR = (1.0, 0.4, 0.3)

    try:
        for p_idx, pose_deg in enumerate(SIM_POSES_DEG):
            scene.set_joints(pose_deg)
            real_obs = env.move_to_joint(pose_deg)

            plotter = pv.Plotter(title=f"pose {p_idx}: {pose_deg}")
            plotter.add_axes_at_origin(line_width=2)

            for rs_id, p in cam_params.items():
                sim_rgb, sim_depth = scene.render_rgbd(rs_id)
                sim_pts, _ = scene.lift_to_world(rs_id, sim_depth, sim_rgb)

                rs_rgb_bgr = real_obs[rs_id]["color"]
                rs_depth = real_obs[rs_id]["depth"]
                rs_pts, _ = depth_to_world_points(
                    rs_depth, rs_rgb_bgr, p["intrinsic"], p["extrinsic"],
                )

                plotter.add_mesh(
                    pv.PolyData(sim_pts),
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
                    f"[pose{p_idx} {rs_id}] sim pts={len(sim_pts)}  real pts={len(rs_pts)}"
                )

            plotter.add_legend()
            print(f"[pose{p_idx}] close the window to advance to the next pose")
            plotter.show()
    finally:
        env.close()


if __name__ == "__main__":
    test_pointcloud_alignment()
