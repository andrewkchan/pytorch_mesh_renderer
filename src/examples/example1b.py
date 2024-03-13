"""
Example 1b: Rendering a teapot from arbitrary angle with the soft rasterizer.
"""

import os
import argparse

import torch
import numpy as np
from skimage import io

from .. import soft_mesh_renderer as smr
from ..common import obj_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1b.png'))
    args = parser.parse_args()

    # load obj file
    vertices, triangles, normals = obj_utils.load_obj(args.filename_input)
    vertices = vertices[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]
    # TODO why are triangles not batched?
    normals = normals[None,:,:] # [num_vertices, 3] -> [batch_size=1, num_vertices, 3]

    # camera position:
    eye = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
    center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    world_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    # create a diffuse colors tensor coloring all vertices white
    vertex_diffuse_colors = torch.ones_like(vertices, dtype=torch.float32)

    light_positions = torch.tensor([[[0.0, 3.0, 0.0]]], dtype=torch.float32)
    light_intensities = torch.ones([1, 1], dtype=torch.float32)

    image_width = 100
    image_height = 100

    render = smr.render(
        vertices,
        triangles,
        vertex_diffuse_colors,
        eye,
        center,
        world_up,
        light_positions,
        light_intensities,
        image_width,
        image_height
    )
    render = torch.reshape(render, [image_height, image_width, 4])
    result_image = render.numpy()
    result_image = np.clip(result_image, 0., 1.).copy(order="C")

    io.imsave(args.filename_output, (result_image * 255.0).astype(np.uint8))
