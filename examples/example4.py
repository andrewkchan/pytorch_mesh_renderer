"""
Example 4: Optimizing camera angles to reconstruct a teapot render.

This example doesn't converge with the barycentric-based differentiable renderer.
"""

import os
import argparse

import torch
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt

import mesh_renderer as mr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-t', '--filename_target', type=str, default=os.path.join(data_dir, 'example4_target.png'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example4.mp4'))
    args = parser.parse_args()

    # load obj file
    vertices, triangles, normals = mr.load_obj(args.filename_input)
    vertices = vertices[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]
    # TODO why are triangles not batched?
    normals = normals[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]

    image_width = 640
    image_height = 480

    target_render = torch.tensor(
        io.imread(args.filename_target).astype(float) / 255.0
    )[None,:,:,:] # [image_width, image_height, 4] -> [batch_size=1, image_width, image_height, 4]

    # create a diffuse colors tensor coloring all vertices white
    vertex_diffuse_colors = torch.ones_like(vertices, dtype=torch.float32)

    light_positions = torch.tensor([[[0.0, 3.0, 0.0]]], dtype=torch.float32)
    light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

    # camera position:
    # initial_eye = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float32)
    # initial_world_up = torch.tensor([0.0, 3.0, -2.0], dtype=torch.float32)
    initial_eye = torch.tensor([0.0, 3.0, 3.0], dtype=torch.float32)
    initial_world_up = torch.tensor([0.0, 3.0, -3.0], dtype=torch.float32)
    eye = torch.tensor(initial_eye[None,:], dtype=torch.float32, requires_grad=True)
    camera_euler_angles = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=False)

    writer = imageio.get_writer(args.filename_output, fps=20)
    optimizer = torch.optim.SGD([eye, camera_euler_angles], 0.7, 0.1)
    def stepfn():
        optimizer.zero_grad()
        
        camera_euler_transforms = mr.euler_matrices(camera_euler_angles)[0, :3, :3] # [3, 3]
        forward = torch.reshape(torch.matmul(-initial_eye, camera_euler_transforms.T), [1, 3])
        world_up = torch.reshape(torch.matmul(initial_world_up, camera_euler_transforms.T), [1, 3])
        center = eye + forward
        render = mr.mesh_renderer(
            vertices, triangles, normals,
            vertex_diffuse_colors, eye, center, world_up, light_positions,
            light_intensities, image_width, image_height)
        
        # write to GIF output
        frame = render[0].detach().numpy() # [image_height, image_width, 4]
        # black background
        frame = np.concatenate([
            frame[:,:,:3]*frame[:,:,3][:,:,None], 
            np.ones([image_height, image_width, 1], dtype=np.float32)
        ], axis=-1)
        writer.append_data((255*frame).astype(np.uint8))

        loss = torch.mean(torch.abs(render - target_render))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([eye, center, world_up], 1.0)
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


