import unittest

import numpy as np
import pyrealsense2 as rs

from digital_twin_camera.cameras import pinhole_remap


class CameraRectificationTest(unittest.TestCase):
    def test_remap_round_trip_through_sdk(self):
        intrinsic = rs.intrinsics()
        intrinsic.width, intrinsic.height = 1280, 720
        intrinsic.fx, intrinsic.fy = 647.0814209, 646.1427002
        intrinsic.ppx, intrinsic.ppy = 635.53320312, 350.85418701
        intrinsic.coeffs = [-0.053503, 0.054778, 0.000160613, 0.000853641, -0.0171287]
        for model in (
            rs.distortion.none,
            rs.distortion.brown_conrady,
            rs.distortion.inverse_brown_conrady,
        ):
            with self.subTest(model=model):
                intrinsic.model = model
                map_x, map_y = pinhole_remap(intrinsic)
                for v in range(0, 720, 37):
                    for u in range(0, 1280, 41):
                        ray = rs.rs2_deproject_pixel_to_point(
                            intrinsic,
                            [float(map_x[v, u]), float(map_y[v, u])],
                            1.0,
                        )
                        pixel = [
                            ray[0] * intrinsic.fx + intrinsic.ppx,
                            ray[1] * intrinsic.fy + intrinsic.ppy,
                        ]
                        np.testing.assert_allclose(pixel, [u, v], atol=0.001, rtol=0)

    def test_unsupported_model_is_rejected(self):
        intrinsic = rs.intrinsics()
        intrinsic.model = rs.distortion.kannala_brandt4
        with self.assertRaisesRegex(ValueError, "Unsupported color distortion"):
            pinhole_remap(intrinsic)


if __name__ == "__main__":
    unittest.main()
