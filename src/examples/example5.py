"""
Example 4: Optimizing rotation of a cube.
"""

import os
import argparse

import torch
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt

import mesh_renderer as mr
from ..common import camera_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--filename_target', type=str, default=os.path.join(data_dir, '../mesh_renderer/test_data/Gray_Cube_0.png'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example5.mp4'))
    args = parser.parse_args()

    image_width = 640
    image_height = 480

    # Set up a basic cube centered at the origin, with vertex normals pointing
    # outwards along the line from the origin to the cube vertices:
    cube_vertices = torch.tensor(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
        [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=torch.float32)
    cube_normals = torch.nn.functional.normalize(cube_vertices, dim=1, p=2)
    cube_triangles = torch.tensor(
        [[0, 1, 2], [2, 3, 0], [3, 2, 6], [6, 7, 3], [7, 6, 5], [5, 4, 7],
            [4, 5, 1], [1, 0, 4], [5, 6, 2], [2, 1, 5], [7, 4, 0], [0, 3, 7]],
        dtype=torch.int32)

    initial_euler_angles = [[0.0, 0.0, 0.0]]
    euler_angles = torch.tensor(initial_euler_angles, requires_grad=True)

    def render_cube_with_rotation(input_euler_angles):
        model_rotation = camera_utils.euler_matrices(input_euler_angles)[0, :3, :3] # [3, 3]

        vertices_world_space = torch.reshape(
            torch.matmul(cube_vertices, model_rotation.T),
            [1, 8, 3])

        normals_world_space = torch.reshape(
            torch.matmul(cube_normals, model_rotation.T),
            [1, 8, 3])

        # camera position:
        eye = torch.tensor([[0.0, 0.0, 6.0]], dtype=torch.float32)
        center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        world_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

        vertex_diffuse_colors = torch.ones_like(vertices_world_space, dtype=torch.float32)
        light_positions = torch.reshape(eye, [1, 1, 3])
        light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

        render = mr.render(
            vertices_world_space, cube_triangles, normals_world_space,
            vertex_diffuse_colors, eye, center, world_up, light_positions,
            light_intensities, image_width, image_height)
        render = torch.reshape(render, [image_height, image_width, 4])
        return render

    target_render = torch.tensor(
        io.imread(args.filename_target).astype(float) / 255.0
    ) # [image_width, image_height, 4]

    writer = imageio.get_writer(args.filename_output, fps=20)
    optimizer = torch.optim.SGD([euler_angles], 0.7, 0.1)
    def stepfn():
        optimizer.zero_grad()
        render = render_cube_with_rotation(euler_angles)

        # write to GIF output
        frame = render.detach().numpy() # [image_height, image_width, 4]
        # black background
        frame = np.concatenate([
            frame[:,:,:3]*frame[:,:,3][:,:,None],
            np.ones([image_height, image_width, 1], dtype=np.float32)
        ], axis=-1)
        writer.append_data((255*frame).astype(np.uint8))

        loss = torch.mean(torch.abs(render - target_render))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([euler_angles], 1.0)
        return loss

    epochs = 50
    loss_points = []
    for e in range(epochs):
        print("step {} of {}".format(e, epochs))
        loss = optimizer.step(stepfn)
        loss_points.append(float(loss))

    writer.close()

    x = np.arange(0, epochs, 1)
    y = np.array(loss_points)
    plt.plot(x, y)
    plt.show()
