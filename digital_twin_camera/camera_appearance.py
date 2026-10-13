"""Apply the configured appearance mode to the sensor producing RGB."""

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_camera_appearance() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if "camera_appearance" not in config:
        return {"mode": "auto"}
    return config["camera_appearance"]


def apply_camera_appearance(profile, appearance: dict) -> dict[str, float]:
    import pyrealsense2 as rs

    mode = appearance["mode"]
    if mode not in ("auto", "align"):
        raise ValueError(f"Unknown camera appearance mode: {mode!r}")
    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    sensor = next(
        sensor for sensor in device.query_sensors()
        if any(stream.stream_type() == rs.stream.color
               for stream in sensor.get_stream_profiles())
    )
    settings = {
        "enable_auto_exposure": float(mode == "auto"),
        "enable_auto_white_balance": float(mode == "auto"),
    }
    if mode == "align":
        model = device.get_info(rs.camera_info.name).split()[-1].lower()
        exposure_unit_us = {"d455": 100, "d405": 1}[model]
        settings.update(appearance["align"]["shared"])
        settings["exposure"] = appearance["align"]["exposure_us"] / exposure_unit_us
        settings["gain"] = appearance["align"]["gain"][model]

    # Validate the complete request before changing the hardware.
    for name, value in settings.items():
        option = getattr(rs.option, name)
        if not sensor.supports(option):
            raise ValueError(f"RealSense {serial} does not support {name}")
        limits = sensor.get_option_range(option)
        if not limits.min <= value <= limits.max:
            raise ValueError(f"RealSense {serial}: {name}={value} outside {limits}")
        if limits.step and abs((value - limits.min) / limits.step -
                               round((value - limits.min) / limits.step)) > 1e-6:
            raise ValueError(f"RealSense {serial}: {name}={value} violates step {limits.step}")

    for name, value in settings.items():
        sensor.set_option(getattr(rs.option, name), value)
    effective = {
        name: sensor.get_option(getattr(rs.option, name)) for name in settings
    }
    for name, value in settings.items():
        if abs(effective[name] - value) > 1e-6:
            raise RuntimeError(f"RealSense {serial}: requested {name}={value}, read {effective[name]}")
    print(f"RealSense {serial} appearance={mode}: {effective}", flush=True)
    return effective
