import os
import unittest

import numpy as np
import torch

import camera_utils
import mesh_renderer
import test_utils

class RenderTest(unittest.TestCase):
    def setUp(self):
        self.test_data_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test_data'
        )

        # Set up a basic cube centered at the origin, with vertex normals pointing
        # outwards along the line from the origin to the cube vertices:
        self.cube_vertices = torch.tensor(
            [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
            [1, -1, -1], [1, 1, -1], [1, 1, 1]],
            dtype=torch.float32)
        self.cube_normals = torch.nn.functional.normalize(self.cube_vertices, dim=1, p=2)
        self.cube_triangles = torch.tensor(
            [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
             [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
            dtype=torch.int32)

    def testRendersSimpleCube(self):
        """Renders a simple cube to test the full forward pass.

        Verifies the functionality of both the custom kernel and the python wrapper.
        """

        model_transforms = camera_utils.euler_matrices(
            torch.tensor([[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]]))[:, :3, :3]

        vertices_world_space = torch.matmul(
            torch.stack([self.cube_vertices, self.cube_vertices]),
            model_transforms.transpose(1, 2)
        )

        normals_world_space = torch.matmul(
            torch.stack([self.cube_normals, self.cube_normals]),
            model_transforms.transpose(1, 2)
        )

        # camera position:
        eye = torch.tensor(2 * [[0.0, 0.0, 6.0]], dtype=torch.float32)
        center = torch.tensor(2 * [[0.0, 0.0, 0.0]], dtype=torch.float32)
        world_up = torch.tensor(2 * [[0.0, 1.0, 0.0]], dtype=torch.float32)
        image_width = 640
        image_height = 480
        light_positions = torch.tensor([[[0.0, 0.0, 6.0]], [[0.0, 0.0, 6.0]]])
        light_intensities = torch.ones([2, 1, 3], dtype=torch.float32)
        vertex_diffuse_colors = torch.ones_like(vertices_world_space, dtype=torch.float32)

        images = mesh_renderer.mesh_renderer(
            vertices_world_space, self.cube_triangles, normals_world_space,
            vertex_diffuse_colors, eye, center, world_up, light_positions,
            light_intensities, image_width, image_height
        )

        for image_id in range(images.shape[0]):
            target_image_name = 'Gray_Cube_%i.png' % image_id
            baseline_image_path = os.path.join(self.test_data_directory,
                                            target_image_name)
            test_utils.expect_image_file_and_render_are_near(
                self, baseline_image_path, images[image_id, :, :, :])

    def testComplexShading(self):
        """Tests specular highlights, colors, and multiple lights per image."""
        # rotate the cube for the test:
        model_transforms = camera_utils.euler_matrices(
            torch.tensor([[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]]))[:, :3, :3]

        vertices_world_space = torch.matmul(
            torch.stack([self.cube_vertices, self.cube_vertices]),
            model_transforms.transpose(1, 2)
        )

        normals_world_space = torch.matmul(
            torch.stack([self.cube_normals, self.cube_normals]),
            model_transforms.transpose(1, 2)
        )

        # camera position:
        eye = torch.tensor([[0.0, 0.0, 6.0], [0., 0.2, 18.0]], dtype=torch.float32)
        center = torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.1]], dtype=torch.float32)
        world_up = torch.tensor(
            [[0.0, 1.0, 0.0], [0.1, 1.0, 0.15]], dtype=torch.float32)
        fov_y = torch.tensor([40., 13.3], dtype=torch.float32)
        near_clip = torch.tensor(0.1, dtype=torch.float32)
        far_clip = torch.tensor(25.0, dtype=torch.float32)
        image_width = 640
        image_height = 480
        light_positions = torch.tensor([[[0.0, 0.0, 6.0], [1.0, 2.0, 6.0]],
                                    [[0.0, -2.0, 4.0], [1.0, 3.0, 4.0]]])
        light_intensities = torch.tensor(
            [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[2.0, 0.0, 1.0], [0.0, 2.0,
                                                                    1.0]]],
            dtype=torch.float32)
        vertex_diffuse_colors = torch.tensor(2*[[[1.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0],
                                                [1.0, 1.0, 1.0],
                                                [1.0, 1.0, 0.0],
                                                [1.0, 0.0, 1.0],
                                                [0.0, 1.0, 1.0],
                                                [0.5, 0.5, 0.5]]],
                                            dtype=torch.float32)
        vertex_specular_colors = torch.tensor(2*[[[0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0],
                                                [1.0, 1.0, 1.0],
                                                [1.0, 1.0, 0.0],
                                                [1.0, 0.0, 1.0],
                                                [0.0, 1.0, 1.0],
                                                [0.5, 0.5, 0.5],
                                                [1.0, 0.0, 0.0]]],
                                            dtype=torch.float32)
        shininess_coefficients = 6.0 * torch.ones([2, 8], dtype=torch.float32)
        ambient_color = torch.tensor(
            [[0., 0., 0.], [0.1, 0.1, 0.2]], dtype=torch.float32)
        renders = mesh_renderer.mesh_renderer(
            vertices_world_space, self.cube_triangles, normals_world_space,
            vertex_diffuse_colors, eye, center, world_up, light_positions,
            light_intensities, image_width, image_height, vertex_specular_colors,
            shininess_coefficients, ambient_color, fov_y, near_clip, far_clip)
        tonemapped_renders = torch.cat(
            [
                mesh_renderer.tone_mapper(renders[:, :, :, 0:3], 0.7),
                renders[:, :, :, 3:4]
            ],
            dim=3)

        # Check that shininess coefficient broadcasting works by also rendering
        # with a scalar shininess coefficient, and ensuring the result is identical:
        broadcasted_renders = mesh_renderer.mesh_renderer(
            vertices_world_space, self.cube_triangles, normals_world_space,
            vertex_diffuse_colors, eye, center, world_up, light_positions,
            light_intensities, image_width, image_height, vertex_specular_colors,
            6.0, ambient_color, fov_y, near_clip, far_clip)
        tonemapped_broadcasted_renders = torch.cat(
            [
                mesh_renderer.tone_mapper(broadcasted_renders[:, :, :, 0:3], 0.7),
                broadcasted_renders[:, :, :, 3:4]
            ],
            dim=3)

    def testFullRenderGradientComputation(self):
        """Verifies the Jacobian matrix for the entire renderer.

        This ensures correct gradients are propagated backwards through the entire
        process, not just through the rasterization kernel. Uses the simple cube
        forward pass.
        """
        image_height = 21
        image_width = 28

        def render_cube_vertices(cube_vertices):
            # rotate the cube for the test:
            model_transforms = camera_utils.euler_matrices(
                torch.tensor([[-20.0, 0.0, 60.0], [45.0, 60.0, 0.0]]))[:, :3, :3]

            vertices_world_space = torch.matmul(
                torch.stack([cube_vertices, cube_vertices]),
                model_transforms.transpose(1, 2))

            normals_world_space = torch.matmul(
                torch.stack([self.cube_normals, self.cube_normals]),
                model_transforms.transpose(1, 2))

            # camera position:
            eye = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float32)
            center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

            # Scene has a single light from the viewer's eye.
            light_positions = torch.unsqueeze(torch.stack([eye, eye], dim=0), dim=1)
            light_intensities = torch.ones([2, 1, 3], dtype=torch.float32)

            vertex_diffuse_colors = torch.ones_like(vertices_world_space, dtype=torch.float32)

            rendered = mesh_renderer.mesh_renderer(
                vertices_world_space, self.cube_triangles, normals_world_space,
                vertex_diffuse_colors, eye, center, world_up, light_positions,
                light_intensities, image_width, image_height)
            return rendered

        test_cube_vertices = torch.tensor(self.cube_vertices, requires_grad=True)
        analytical = test_utils.get_analytical_jacobian(
            test_cube_vertices, render_cube_vertices(test_cube_vertices))
        numerical = test_utils.get_numerical_jacobian(
            render_cube_vertices, test_cube_vertices, eps=1e-3)
        jacobians_match = (
            test_utils.check_jacobians_are_nearly_equal(
                analytical, numerical, 0.01, 0.01))
        self.assertTrue(
            jacobians_match,
            "Analytical and numerical jacobians have too many relative or "
            "absolute outliers")

    def testThatCubeRotates(self):
        """Optimize a simple cube's rotation using pixel loss.

        The rotation is represented as static-basis euler angles. This test checks
        that the computed gradients are useful.
        """
        image_height = 480
        image_width = 640
        initial_euler_angles = [[0.0, 0.0, 0.0]]
        euler_angles = torch.tensor(initial_euler_angles, requires_grad=True)

        def render_cube_with_rotation(input_euler_angles):
            model_rotation = camera_utils.euler_matrices(input_euler_angles)[0, :3, :3] # [3, 3]

            vertices_world_space = torch.reshape(
                torch.matmul(self.cube_vertices, model_rotation.T),
                [1, 8, 3])

            normals_world_space = torch.reshape(
                torch.matmul(self.cube_normals, model_rotation.T),
                [1, 8, 3])

            # camera position:
            eye = torch.tensor([[0.0, 0.0, 6.0]], dtype=torch.float32)
            center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            world_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

            vertex_diffuse_colors = torch.ones_like(vertices_world_space, dtype=torch.float32)
            light_positions = torch.reshape(eye, [1, 1, 3])
            light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

            render = mesh_renderer.mesh_renderer(
                vertices_world_space, self.cube_triangles, normals_world_space,
                vertex_diffuse_colors, eye, center, world_up, light_positions,
                light_intensities, image_width, image_height)
            render = torch.reshape(render, [image_height, image_width, 4])
            return render

        # Pick the desired cube rotation for the test:
        target_euler_angles = torch.tensor([[-20.0, 0.0, 60.0]])
        desired_render = render_cube_with_rotation(target_euler_angles)

        optimizer = torch.optim.SGD([euler_angles], 0.7, 0.1)
        def stepfn():
            optimizer.zero_grad()
            render = render_cube_with_rotation(euler_angles)
            loss = torch.mean(torch.abs(render - desired_render))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([euler_angles], 1.0)
            return loss

        for _ in range(35):
            optimizer.step(stepfn)

        final_render = render_cube_with_rotation(euler_angles)
        desired_render = render_cube_with_rotation(target_euler_angles) # sanity check re-rendering target angles is the same

        target_image_name = 'Gray_Cube_0.png'
        baseline_image_path = os.path.join(self.test_data_directory,
                                            target_image_name)
        test_utils.expect_image_file_and_render_are_near(
            self, baseline_image_path, desired_render)
        test_utils.expect_image_file_and_render_are_near(
            self,
            baseline_image_path,
            final_render.detach(),
            max_outlier_fraction=0.01,
            pixel_error_threshold=0.04)


if __name__ == "__main__":
    unittest.main()