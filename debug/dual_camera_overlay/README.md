# Mug/rack dual-camera comparison

From the calibration repository root, in the `calibrate` environment:

```bash
python -m debug.tmp_dual_camera_overlay
```

This regenerates `scene.xml`, an isolated robot XML, and mug/rack XMLs from
Datagen's measured `hang_mug` assets. Meshes and textures are referenced from
the sibling `Codegen_Datagen` repository. Existing calibration scene XMLs are
not overwritten. The saved D405 pose must have been recalibrated after the
distortion correction.

The window has one row per camera: **MuJoCo | real | overlay**. D455 uses its
existing raw-image calibration workflow at 640×480; D405 captures at 1280×720
and undistorts before intrinsic warping. Both render at 1280×720, and display
panels are reduced to fit the window. Capture runs concurrently in the two
devices; frame reads are sequential, without hardware synchronization.

- **U:** toggle D405 undistortion. The calibrated pose stays fixed, so this
  isolates the image correction, not the entire old calibration pipeline.
- **Simulation %:** change the blend amount.
- **S:** save full-resolution images, both D405 correction variants, the
  combined window, and pose/intrinsic metadata under `capture/`.
- **Q / Esc:** exit.

Place the real objects at the simulated board-world poses, or supply measured
poses with `--mug X Y Z YAW` and `--rack X Y Z YAW`. XYZ are meters; yaw is
degrees about world +Z. Defaults from Datagen's example are mug
`0.23 -0.30 0 180` and rack `0.32 0 0 -80`. These are asset-body origins, not
necessarily geometric centers. At the default wrist pose the objects are
partially clipped; choose physical placements within the wrist view before
judging alignment. The board origin and orientation must remain those used
for both camera calibrations.

The physical robot must remain stationary at the supplied `--joints` (seven
degrees, default `0 -45 0 30 0 75 0`) with the same gripper opening as the
existing wrist scene. The script does not read or move the robot. The saved
wrist calibration must correspond to this joint configuration.

Headless validation and capture:

```bash
MUJOCO_GL=egl python -m debug.tmp_dual_camera_overlay --render-only
MUJOCO_GL=egl python -m debug.tmp_dual_camera_overlay --snapshot
```

Render-only saves both simulated views under `renders/`. Snapshot saves one
live comparison and exits. These output images are ignored by Git.
