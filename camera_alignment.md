# D455 / D405 appearance alignment plan

## Goal

Make the D455 and D405 produce similar, stable brightness and color, with an explicit choice between automatic camera adjustment and a fixed alignment profile. This is appearance alignment; it does not change camera poses, intrinsics or geometric calibration.

The current capture paths configure streams but do not explicitly configure exposure or color controls. Both paths must use the same appearance policy:

- `digital_twin_camera/camera.py`: grouped camera worker used by the robot environment.
- `digital_twin_camera/cameras.py`: RGB capture used by calibration and overlay tools.

Identical numeric settings do not guarantee identical appearance. The D405 derives RGB from its depth imager, so available controls and their sensor ownership differ. Discover actual device capabilities before choosing values. See the [RealSense D405 documentation](https://www.realsenseai.com/product-family/d405-series/).

## Mode behavior

| Mode | Behavior | Intended use |
| --- | --- | --- |
| `auto` | Explicitly enable supported automatic exposure and automatic white balance. Do not apply manual exposure/gain/white-balance settings or alignment color correction. | Changing illumination and normal automatic capture. |
| `align` | Apply a measured, fixed profile for each camera. Disable automatic controls where supported before setting manual values. Apply optional measured color correction only when explicitly configured. | Stable illumination and repeatable appearance across cameras. |

`align` means using a saved alignment profile at startup. Profile measurement is a separate tuning operation, rather than continuous adjustment during capture. Unsupported automatic controls must be reported; the mode cannot promise control over settings the hardware does not expose.

## Configuration

Current implementation: `auto` is the code default when the appearance section is absent. This workspace selects `camera_appearance.mode: align` in `config.yaml` for current use. Both capture paths apply it before warmup. To return to automatic adjustment, set that value to `auto`.

The initial alignment baseline uses 16,600 microseconds of exposure, 4600 K white balance, and identical supported brightness, contrast, saturation, gamma and sharpness settings. Gain retains each model's sensor baseline (D455: 64; D405: 16) because equal gain numbers do not describe equal amplification across models. This is a fixed starting profile, not a completed color-target calibration. SDK exposure conversion is D455 RGB: 100 microseconds per unit; D405: 1 microsecond per unit. See [SDK exposure units discussion](https://github.com/realsenseai/librealsense/issues/8243).

On-device verification confirmed setting readback, color capture and switching between `auto` and `align` on connected D455 `239222303153` and D405 `262122271915`. The configured D405 serial is currently `130322271320`; resolving that device mismatch is required to capture the configured pair. Target-based visual tuning, optional correction and comparison tooling below remain planned work.

Add one `camera_appearance` section to `config.yaml`, containing:

- `mode`: `auto` or `align`; the code default is `auto`, with `align` selected for the current workspace.
- Shared capture policy: frame rate and supported anti-flicker setting, with its physical meaning documented.
- Explicit D455 and D405 alignment profiles, with manual exposure expressed in a documented physical unit, gain in device units, and supported white balance and image controls.
- Optional color correction parameters per camera, disabled initially.

Store only measured, supported settings in the final profiles. Do not invent initial exposure values or copy raw SDK exposure numbers between models without verifying their units. Use matching physical exposure time and common settings where feasible, then allow measured differences to achieve similar appearance.

Keep configuration immutable. Resolve the selected mode into local settings and pass the appearance configuration explicitly through both capture paths. Do not pass a parent configuration together with its child configuration.

## Implementation steps

1. **Inspect hardware capabilities.** Enumerate the configured serials and active sensors. Record supported controls, current values, ranges, increments, units, firmware and stream profiles. Locate the sensor that actually controls each color stream rather than assuming every device has a separate RGB sensor. Inspect exposure, gain, auto exposure, white balance, auto white balance, brightness, contrast, saturation, gamma, sharpness and anti-flicker controls.

2. **Implement one shared settings application function.** Call it after pipeline startup and before warmup in both capture paths. Apply automatic/manual switches before dependent values. Validate requested settings against device capabilities and fail clearly for unsupported or out-of-range explicit settings; do not silently skip them. Report inherently unavailable controls during discovery. Read settings back and report the effective values with their camera serials.

3. **Wire the mode through existing entry points.** Update robot environment, calibration and overlay camera construction to consume the same configured mode. Keep existing stream geometry and image conventions; both paths must continue returning BGR images. Select the mode in configuration first; avoid adding redundant command-line switches unless needed by existing workflows.

4. **Measure an alignment profile.** Under steady, diffuse illumination, capture a common gray/color target with both cameras. Choose supported stream profiles and a common frame rate for the comparison. Start with matching physical exposure time where possible, tune gain for brightness, and tune supported white balance/color controls for neutral colors. Account for camera-specific ranges and settings. Save final values only after checking them on the actual devices.

5. **Compare the same target regions.** Provide a small capture/comparison utility with side-by-side images and target-region brightness, channel balance, clipping and temporal variation measurements. Use corresponding target patches rather than whole-image averages because the cameras have different viewpoints. Establish acceptance thresholds from baseline measurements before final tuning.

6. **Add correction only if measurements justify it.** If fixed hardware settings leave a visible color mismatch, fit a simple per-camera color correction from the target patches. Keep correction computation pure, apply it consistently in both capture paths in `align` mode, and preserve access to uncorrected images for diagnosis. Do not use arbitrary per-frame normalization, which would undo repeatability.

7. **Document and verify.** Document mode selection, measured profiles, supported limitations and retuning after lighting changes. Verify startup readback, mode switching through camera restart, and appearance stability in both capture paths. Check D405 depth quality when modifying exposure/gain because those controls may also affect depth acquisition.

## Lighting

Camera software cannot change physical room lighting without a separate lighting controller. Use consistent diffuse lighting, avoid mixed color temperatures, and confirm whether the lamps flicker. Select anti-flicker settings based on the actual lighting supply and supported camera controls. Retune the saved alignment profile when illumination changes materially. MuJoCo lighting is a separate simulation setting and is outside this real-camera alignment change.

## Validation and completion

- Focused tests verify mode selection, automatic controls preceding manual settings, exposure-unit conversion, clear errors for invalid hardware settings, and consistent application in both capture paths.
- On-device checks confirm effective settings after warmup and repeatability after restart.
- Target captures demonstrate improved brightness/color agreement and stable appearance over multiple frames under fixed illumination.
- Existing calibration/overlay workflows still use the same image geometry and BGR convention; depth remains usable.
- `auto` restores supported automatic controls and bypasses optional alignment correction.

The code can be implemented without selecting guessed profile values. Final visual acceptance and saved alignment values require access to both cameras and the shared target under the intended lighting.
