from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np
import torch

import camera_utils
import rasterize_triangles
import test_utils


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.test_data_directory = "mesh_renderer/test_data/"

        self.cube_vertex_positions = torch.tensor(
            [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
             [1, -1, -1], [1, 1, -1], [1, 1, 1]],
            dtype=torch.float32)
        self.cube_triangles = torch.tensor(
            [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
             [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
            dtype=torch.int32)

        self.image_width = 640
        self.image_height = 480

        self.perspective = camera_utils.perspective(
            self.image_width / self.image_height,
            torch.tensor([40.0]), torch.tensor([0.01]),
            torch.tensor([10.0]))

    def runTriangleTest(self, w_vector, target_image_name):
        """Directly renders a rasterized triangle's barycentric coordinates.

        Tests only the kernel (rasterize_triangles_module).

        Args:
            w_vector: 3-vector of w components to scale triangle vertices.
            target_image_name: image file name to compare result against.
        """
        clip_init = np.array(
            [
                [-0.5, -0.5, 0.8, 1.0],
                [0.0, 0.5, 0.3, 1.0],
                [0.5, -0.5, 0.3, 1.0]
            ], dtype=np.float32)
        clip_init = clip_init * np.reshape(
            np.array(w_vector, dtype=np.float32), [3, 1])

        clip_coordinates = torch.tensor(clip_init)
        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        image, _, _ = (
            rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
                clip_coordinates,
                triangles,
                self.image_width,
                self.image_height))
        image = torch.cat(
            [rendered_coordinates,
             torch.ones([self.image_height, self.image_width, 1])], axis=2)
        baseline_image_path = os.path.join(self.test_data_directory,
                                           target_image_name)
        test_utils.expect_image_file_and_render_are_near(
            self, baseline_image_path, image)

    def testRendersSimpleTriangle(self):
        self.runTriangleTest((1.0, 1.0, 1.0), "Simple_Triangle.png")

    def testRendersPerspectiveCorrectTriangle(self):
        self.runTriangleTest((0.2, 0.5, 2.0),
                             "Perspective_Corrected_Triangle.png")

    def testRendersTwoCubesInBatch(self):
        """Renders a simple cube in two viewpoints to test the python wrapper.
        """

        vertex_rgb = (self.cube_vertex_positions * 0.5 + 0.5)
        vertex_rgba = torch.cat([vertex_rgb, torch.ones([8, 1])], axis=1)

        center = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        world_up = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        look_at_1 = camera_utils.look_at(
            torch.tensor([[2, 3, 6]], dtype=torch.float32),
            center,
            world_up)
        look_at_2 = camera_utils.look_at(
            torch.tensor([[-3, 1, 6]], dtype=torch.float32),
            center,
            world_up)
        projection_1 = torch.matmul(self.perspective, look_at_1)
        projection_2 = torch.matmul(self.perspective, look_at_2)
        projection = torch.cat([projection_1, projection_2], axis=0)
        background_value = [0., 0., 0., 0.]

        rendered = rasterize_triangles.rasterize(
            torch.stack([self.cube_vertex_positions,
                         self.cube_vertex_positions]),
            torch.stack([vertex_rgba, vertex_rgba]),
            self.cube_triangles,
            projection,
            self.image_width,
            self.image_height,
            background_value)

        for i in (0, 1):
            image = rendered[i, :, :, :]
            baseline_image_name = "Unlit_Cube_{}.png".format(i)
            baseline_image_path = os.path.join(self.test_data_directory,
                                               baseline_image_name)
            test_utils.expect_image_file_and_render_are_near(
                self, baseline_image_path, image)

    def testSimpleTriangleGradientComputation(self):
        """Verify the Jacobian matrix for a single pixel.

        The pixel is in the center of a triangle facing the camera. This makes
        it easy to check which entries of the Jacobian might not make sense
        without worrying about corner cases.
        """
        test_pixel_x = 325
        test_pixel_y = 245

        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        def rasterize_test_pixels(clip_coordinates):
            barycentric_coordinates, _, _ = (
                rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
                    clip_coordinates,
                    triangles,
                    self.image_width,
                    self.image_height))

            pixels_to_compare = barycentric_coordinates[
                test_pixel_y: test_pixel_y + 1, test_pixel_x:test_pixel_x + 1, :]
            return pixels_to_compare

        test_clip_coordinates = np.array(
            [[-0.5, -0.5, 0.8, 1.0],
             [0.0, 0.5, 0.3, 1.0],
             [0.5, -0.5, 0.3, 1.0]],
            dtype=np.float32,
            requires_grad=True)
        jacobians_match = torch.autograd.gradcheck(
            rasterize_test_pixels,
            test_clip_coordinates,
            eps=4e-2,
            atol=0.1,
            rtol=0.01)
        self.assertTrue(
            jacobians_match,
            "Analytical and numerical jacobians have too many relative or "
            "absolute outliers")

    def testInternalRenderGradientComputation(self):
        """Isolates and verifies the Jacobian matrix for the custom kernel."""
        image_height = 21
        image_width = 28

        def get_barycentric_coordinates(clip_coordinates):
            barycentric_coordinates, _, _ = (
                rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
                    clip_coordinates,
                    self.cube_triangles,
                    image_width,
                    image_height))
            return barycentric_coordinates

        # Precomputed transformation of the simple cube to normalized device
        # coordinates, in order to isolate the rasterization gradient.
        test_clip_coordinates = np.array(
            [[-0.43889722, -0.53184521, 0.85293502, 1.0],
             [-0.37635487, 0.22206162, 0.90555805, 1.0],
             [-0.22849123, 0.76811147, 0.80993629, 1.0],
             [-0.2805393, -0.14092168, 0.71602166, 1.0],
             [0.18631913, -0.62634289, 0.88603103, 1.0],
             [0.16183566, 0.08129397, 0.93020856, 1.0],
             [0.44147962, 0.53497446, 0.85076219, 1.0],
             [0.53008741, -0.31276882, 0.77620775, 1.0]],
            dtype=np.float32,
            requires_grad=True)
        jacobians_match = torch.autograd.gradcheck(
            get_barycentric_coordinates,
            test_clip_coordinates,
            eps=4e-2,
            atol=0.1,
            rtol=0.01)
        self.assertTrue(
            jacobians_match,
            "Analytical and numerical jacobians have too many relative or "
            "absolute outliers")


if __name__ == "__main__":
    unittest.main()
