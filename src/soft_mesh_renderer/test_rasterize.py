import torch
import numpy as np
import unittest

from .rasterize import rasterize_batch, point_to_segment_nearest
from ..common import debug_utils

class RenderTest(unittest.TestCase):
    def test_point_to_segment_nearest(self):
        """
        Test the point_to_segment_nearest function.
        """
        # Test a point that is closest to the middle of the segment.
        point = torch.tensor([1.0, -1.0], dtype=torch.float32)
        segment = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        expected_nearest = torch.tensor([0.0, 0.0], dtype=torch.float32)
        expected_t = 0.5
        nearest, t = point_to_segment_nearest(point, segment[0], segment[1])
        torch.testing.assert_close(expected_nearest, nearest,
            msg="\n\texpected={}\n\tactual={}".format(expected_nearest, nearest))
        torch.testing.assert_close(expected_t, float(t),
            msg="\n\texpected={}\n\tactual={}".format(expected_t, t))

        # Test a point that is closest to the start of the segment.
        point = torch.tensor([0.0, 0.0], dtype=torch.float32)
        segment = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        expected_nearest = torch.tensor([1.0, 0.0], dtype=torch.float32)
        expected_t = 0.0
        nearest, t = point_to_segment_nearest(point, segment[0], segment[1])
        torch.testing.assert_close(expected_nearest, nearest,
            msg="\n\texpected={}\n\tactual={}".format(expected_nearest, nearest))
        torch.testing.assert_close(expected_t, float(t),
            msg="\n\texpected={}\n\tactual={}".format(expected_t, t))

        # Test a point that is closest to the end of the segment.
        point = torch.tensor([0.0, 1.0], dtype=torch.float32)
        segment = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        expected_nearest = torch.tensor([1.0, 1.0], dtype=torch.float32)
        expected_t = 1.0
        nearest, t = point_to_segment_nearest(point, segment[0], segment[1])
        torch.testing.assert_close(expected_nearest, nearest,
            msg="\n\texpected={}\n\tactual={}".format(expected_nearest, nearest))
        torch.testing.assert_close(expected_t, float(t),
            msg="\n\texpected={}\n\tactual={}".format(expected_t, t))

    def test_single_triangle_forward(self):
        """
        Test the forward rasterization pass by rasterizing a single triangle to a
        small 10x10 image. The image coverage should look like so if hard-rasterized:

        0 0 0 0 0 0 0 0 0 H
        0 0 0 0 0 0 0 0 H 1
        0 0 0 0 0 0 0 H 1 1
        0 0 0 0 0 0 H 1 1 1
        0 0 0 0 0 H 1 1 1 1
        0 0 0 0 H 1 1 1 1 1
        0 0 0 H 1 1 1 1 1 1
        0 0 H 1 1 1 1 1 1 1
        0 H 1 1 1 1 1 1 1 1
        H 1 1 1 1 1 1 1 1 1

        Where 1 indicates full coverage, 0 is no coverage, and H is half-covered
        (for hard-rasterization, this can be either considered in or out).
        """

        # in eye space: z=-1 for all vertices, znear=0.5, zfar=2.5
        clip_space_vertices = torch.tensor(
            [
                [1.0, -1.0, 0.25, 1.0],
                [1.0, 1.0, 0.25, 1.0],
                [-1.0, -1.0, 0.25, 1.0],
            ],
            dtype=torch.float32
        )
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        world_space_vertices = torch.tensor(
            [
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            dtype=torch.float32
        )
        normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32
        )
        diffuse_colors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32
        )
        # one light at effectively infinity
        light_positions = torch.tensor([[0.0, 0.0, 100000.0]], dtype=torch.float32)
        light_intensities = torch.tensor([1.0], dtype=torch.float32)
        image_width, image_height = 10, 10
        sigma_val = 1e-5
        gamma_val = 1e-4

        ##############################################################
        # Case 1: blur radius smaller than a single screen-space pixel
        ##############################################################
        blur_radius = 0.01
        output = rasterize_batch(
            clip_space_vertices,
            triangles,
            ### vertex attributes
            world_space_vertices,
            normals,
            diffuse_colors,
            ### lighting
            light_positions,
            light_intensities,
            ###
            image_width,
            image_height,
            sigma_val,
            gamma_val,
            blur_radius
        )
        expected_red = torch.tensor([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
            [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        ], dtype=torch.float32)
        expected_green = torch.zeros_like(expected_red)
        expected_blue = torch.zeros_like(expected_red)
        expected_alpha = expected_red

        torch.testing.assert_close(output[..., 0], expected_red)
        torch.testing.assert_close(output[..., 1], expected_green)
        torch.testing.assert_close(output[..., 2], expected_blue)
        torch.testing.assert_close(output[..., 3], expected_alpha)
        ##############################################################
        # Case 2: blur radius spans a single screen-space pixel
        ##############################################################
        blur_radius2 = 0.1 * np.sqrt(2.0) + 1e-6
        # This will cause samples blur_radius2 away from a triangle to
        # have a nonzero coverage (1e-3) by the triangle. This will allow
        # the triangle to participate in softmax, which should spike the
        # coverage to 1.0.
        sigma_val2 = -blur_radius2**2 / torch.special.logit(torch.tensor(1e-3))
        output2 = rasterize_batch(
            clip_space_vertices,
            triangles,
            ### vertex attributes
            world_space_vertices,
            normals,
            diffuse_colors,
            ### lighting
            light_positions,
            light_intensities,
            ###
            image_width,
            image_height,
            sigma_val2,
            gamma_val,
            blur_radius2
        )
        expected_red2 = torch.tensor([
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
            [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        ], dtype=torch.float32)
        expected_green2 = torch.zeros_like(expected_red2)
        expected_blue2 = torch.zeros_like(expected_red2)
        expected_alpha2 = expected_red2

        torch.testing.assert_close(output2[..., 0], expected_red2)
        torch.testing.assert_close(output2[..., 1], expected_green2)
        torch.testing.assert_close(output2[..., 2], expected_blue2)
        torch.testing.assert_close(output2[..., 3], expected_alpha2)

if __name__ == "__main__":
    unittest.main()