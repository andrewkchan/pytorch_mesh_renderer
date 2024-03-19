"""
Example 7: Fitting sphere vertices to a cow.
"""

import os
import argparse

import torch
import numpy as np
from skimage import io
import imageio
import matplotlib.pyplot as plt

from .. import mesh_renderer as mr
from .. import soft_mesh_renderer as smr
from ..common import shapes, obj_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', '--target_image1', type=str, default=os.path.join(data_dir, 'example7b_target1.png'))
    parser.add_argument('-t2', '--target_image2', type=str, default=os.path.join(data_dir, 'example7b_target2.png'))
    parser.add_argument('-t3', '--target_image3', type=str, default=os.path.join(data_dir, 'example7b_target3.png'))
    parser.add_argument('-t4', '--target_image4', type=str, default=os.path.join(data_dir, 'example7b_target4.png'))
    parser.add_argument('-o', '--output_model', type=str, default=os.path.join(data_dir, 'example7b.obj'))
    parser.add_argument('-v', '--output_video', type=str, default=os.path.join(data_dir, 'example7b.mp4'))
    args = parser.parse_args()

    # load obj file
    vertices, triangles, _ = shapes.sphere(1.)
    vertices.requires_grad = True

    # camera positions:
    eye = torch.tensor([
        [0.0, 0.0, -3.0],
        [3.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=torch.float32)
    center = torch.zeros_like(eye)
    world_up = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)

    light_positions = torch.tensor([
        [
            [0.0, 0.0, -3.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        [
            [0.0, 0.0, -3.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        [
            [0.0, 0.0, -3.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        [
            [0.0, 0.0, -3.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
    ], dtype=torch.float32)
    light_intensities = torch.ones([4, 3], dtype=torch.float32)

    # Create a diffuse colors tensor coloring all vertices white
    vertex_diffuse_colors = torch.ones([4, vertices.shape[0], 3], dtype=torch.float32)

    image_width = 100
    image_height = 100

    target_render1 = torch.tensor(
        io.imread(args.target_image1).astype(float) / 255.0
    ) # [image_width, image_height, 4]
    target_render2 = torch.tensor(
        io.imread(args.target_image2).astype(float) / 255.0
    ) # [image_width, image_height, 4]
    target_render3 = torch.tensor(
        io.imread(args.target_image3).astype(float) / 255.0
    ) # [image_width, image_height, 4]
    target_render4 = torch.tensor(
        io.imread(args.target_image4).astype(float) / 255.0
    ) # [image_width, image_height, 4]
    target_renders = torch.stack([target_render1, target_render2, target_render3, target_render4], dim=0)

    writer = imageio.get_writer(args.output_video, fps=20)
    optimizer = torch.optim.SGD([vertices], 1.0, 0.1)
    def stepfn():
        optimizer.zero_grad()

        # We need to re-create this tensor from `vertices` each run to
        # ensure it gets changes from optimizer updates.
        batched_vertices = torch.stack([vertices]*4, dim=0)
        batched_renders = smr.render(
            batched_vertices,
            triangles,
            vertex_diffuse_colors,
            eye,
            center,
            world_up,
            light_positions,
            light_intensities,
            image_width,
            image_height,
            sigma_val=1e-4,
            fov_y=60.0,
            blur_radius=0.1
        )
        loss = torch.mean(torch.abs(batched_renders - target_renders))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([vertices], 1.0)

        # write to GIF output
        render = torch.reshape(batched_renders[0], [image_height, image_width, 4])
        frame = render.detach().numpy() # [image_height, image_width, 4]
        # black background
        frame = np.concatenate([
            frame[:,:,:3]*frame[:,:,3][:,:,None],
            np.ones([image_height, image_width, 1], dtype=np.float32)
        ], axis=-1)
        writer.append_data((255*frame).astype(np.uint8))

        return loss

    epochs = 500
    loss_points = []
    for e in range(epochs):
        print("\nstep {} of {}\n".format(e, epochs))
        loss = optimizer.step(stepfn)
        loss_points.append(float(loss))

    writer.close()
    obj_utils.save_obj(args.output_model, vertices, triangles)

    x = np.arange(0, epochs, 1)
    y = np.array(loss_points)
    plt.plot(x, y)
    plt.show()

