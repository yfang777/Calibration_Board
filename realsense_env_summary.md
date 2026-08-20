# RealSense Conda Environment Summary

The `test_mujoco_rgb` script should be run from the `codegen_real` conda
environment, not `calibrator`.

Observed evidence:

- `realsense-viewer` can show the cameras, so the RealSense hardware, USB
  connection, and firmware are basically functional.
- In the `calibrator` environment, `pyrealsense2` can enumerate both D455
  cameras, but the script hit RealSense runtime failures such as:
  - `Frame didn't arrive within 15000`
  - `Couldn't resolve requests`
  - `Device or resource busy`
- Running the same script from `codegen_real` works.

Conclusion:

The failure is environment-specific. The most likely cause is a mismatch or
instability in the RealSense Python/librealsense runtime used by `calibrator`,
not a MuJoCo scene problem and not a calibration-board script logic problem.

Use:

```bash
cd /home/yfang/codegen/Robopil_Calibration_Board_Code
conda activate codegen_real
python -c "from digital_twin_camera.mujoco_sync import test_mujoco_rgb; test_mujoco_rgb()"
```
