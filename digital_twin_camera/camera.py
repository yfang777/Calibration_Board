"""RealSense camera worker process.

Runs one RealSense pipeline per device in its own process and serves
synchronized (color, depth) frames over a multiprocessing Pipe. Depth is
returned in meters and aligned to the color stream.
"""

import multiprocessing as mp

import numpy as np
import pyrealsense2 as rs


class CameraWorker(mp.Process):
    def __init__(self, pipe, rs_id, width=640, height=480, fps=30, warmup_frames=5):
        super().__init__()
        self.pipe = pipe
        self.rs_id = rs_id
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames

    def run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.rs_id)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        align = rs.align(rs.stream.color)

        try:
            profile = pipeline.start(config)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

            for _ in range(self.warmup_frames):
                pipeline.wait_for_frames()

            while True:
                cmd = self.pipe.recv()
                if cmd == "GET":
                    frames = align.process(pipeline.wait_for_frames())
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    color_bgr = np.asanyarray(color_frame.get_data()).copy()
                    depth_raw = np.asanyarray(depth_frame.get_data())
                    depth_m = depth_raw.astype(np.float32) * depth_scale

                    self.pipe.send((self.rs_id, color_bgr, depth_m))
                elif cmd == "STOP":
                    break
        finally:
            pipeline.stop()
