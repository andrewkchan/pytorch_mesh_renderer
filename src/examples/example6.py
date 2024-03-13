"""
Example 6: Optimizing rotation of a teapot.

This example converges for small perturbations in rotation but not larger perturbations
using the barycentric-differentiable renderer.
"""

import os
import argparse

import torch
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt

from .. import mesh_renderer as mr
from ..common import obj_utils
from ..common import camera_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-t', '--filename_target', type=str, default=os.path.join(data_dir, 'example6_target.png'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example6.mp4'))
    args = parser.parse_args()

    image_width = 640
    image_height = 480

    # load obj file
    vertices, triangles, normals = obj_utils.load_obj(args.filename_input)
    vertices = vertices[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]
    # TODO why are triangles not batched?
    normals = normals[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]

    # camera position:
    eye = torch.tensor([[0.0, 3.0, 3.0]], dtype=torch.float32)
    center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    world_up = torch.tensor([0.0, np.cos(-np.pi/4.), np.sin(-np.pi/4.)], dtype=torch.float32)

    vertex_diffuse_colors = torch.ones_like(vertices, dtype=torch.float32)
    light_positions = torch.tensor([[[0.0, 3.0, 0.0]]], dtype=torch.float32)
    light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

    initial_euler_angles = [[np.pi/4., 0., 0.]]
    euler_angles = torch.tensor(initial_euler_angles, requires_grad=True)

    def render_with_rotation(input_euler_angles):
        model_rotation = camera_utils.euler_matrices(input_euler_angles)[0, :3, :3] # [3, 3]

        vertices_world_space = torch.matmul(vertices, model_rotation.T)
        # normals must be transformed using the inverse of the transpose of a matrix M
        normals_world_space = torch.matmul(normals, torch.inverse(model_rotation.T).T)

        render = mr.render(
            vertices_world_space, triangles, normals_world_space,
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
        render = render_with_rotation(euler_angles)

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
