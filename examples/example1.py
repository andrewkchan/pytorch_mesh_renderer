"""
Example 1: Rendering a teapot from arbitrary angle.
"""

import os
import argparse

import torch
import numpy as np
from skimage import io

import mesh_renderer as mr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.png'))
    args = parser.parse_args()

    # load obj file
    vertices, triangles, normals = mr.load_obj(args.filename_input)
    vertices = vertices[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]
    # TODO why are triangles not batched?
    normals = normals[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]

    # camera position:
    eye = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
    center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    world_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    eye = torch.tensor([0.0, 3.0, 3.0], dtype=torch.float32)
    world_up = torch.tensor([0.0, np.cos(-np.pi/4.), np.sin(-np.pi/4.)], dtype=torch.float32)

    # create a diffuse colors tensor coloring all vertices white
    vertex_diffuse_colors = torch.ones_like(vertices, dtype=torch.float32)

    light_positions = torch.tensor([[[0.0, 3.0, 0.0]]], dtype=torch.float32)
    light_intensities = torch.ones([1, 1, 3], dtype=torch.float32)

    image_width = 640
    image_height = 480

    render = mr.mesh_renderer(
        vertices, triangles, normals,
        vertex_diffuse_colors, eye, center, world_up, light_positions,
        light_intensities, image_width, image_height)
    render = torch.reshape(render, [image_height, image_width, 4])
    result_image = render.numpy()
    result_image = np.clip(result_image, 0., 1.).copy(order="C")

    io.imsave(args.filename_output, (result_image * 255.0).astype(np.uint8))
