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

from .. import soft_mesh_renderer as smr
from ..common import shapes, obj_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '.')

# From PyTorch3D:
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/laplacian_matrices.py
#
# Note the laplacian depends only on the topology of a mesh and can be
# considered constant if the topology is fixed.
def compute_laplacian(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    edges = edges.long()
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def compute_edges_list(vertices, faces):
    """
    Computes the edges of a mesh from its vertices and faces.
    Args:
        vertices: tensor of shape (V, 3) containing the vertices of the mesh
        faces: tensor of shape (F, 3) containing the vertex indices of each face
    Returns:
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    """
    faces = faces.to(vertices.device)
    # pyre-fixme[16]: Module `torch` has no attribute `cat`.
    edges = torch.cat(
        [
            faces[:, :2],
            faces[:, 1:],
            faces[:, ::2],
        ]
    )
    edges = edges.view(-1, 2)
    edges = torch.unique(edges, dim=0)
    return edges

def mesh_laplacian_smoothing_loss(vertices, laplacian):
    """
    Computes the uniform weight laplacian smoothing objective for a single mesh (unbatched).
    Args:
        vertices: tensor of shape (V, 3) containing the vertices of the mesh
        laplacian: tensor of shape (V, V) containing the laplacian matrix of the mesh
    Returns:
        loss: the laplacian smoothing loss
    """
    weight = 1.0 / (vertices.shape[0])
    loss = laplacian.mm(vertices)
    loss = loss.norm(dim=1)
    loss = loss * weight
    return loss.sum()

def mesh_edge_loss(vertices, edges):
    """
    Computes the edge length loss for a single mesh (unbatched).
    Args:
        vertices: tensor of shape (V, 3) containing the vertices of the mesh
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        loss: the edge length loss
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    loss = (v0 - v1).norm(dim=1, p=2)
    return loss.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', '--target_image1', type=str, default=os.path.join(data_dir, 'example7b_target1.png'))
    parser.add_argument('-t2', '--target_image2', type=str, default=os.path.join(data_dir, 'example7b_target2.png'))
    parser.add_argument('-t3', '--target_image3', type=str, default=os.path.join(data_dir, 'example7b_target3.png'))
    parser.add_argument('-t4', '--target_image4', type=str, default=os.path.join(data_dir, 'example7b_target4.png'))
    parser.add_argument('-o', '--output_model', type=str, default=os.path.join(data_dir, 'example7b.obj'))
    parser.add_argument('-v', '--output_video', type=str, default=os.path.join(data_dir, 'example7b.mp4'))
    parser.add_argument('-p', '--output_previews_dir', type=str, default=os.path.join(data_dir, 'example7b_previews'))
    args = parser.parse_args()

    # load obj file
    sphere_resolution = 20
    vertices, triangles, _ = shapes.sphere(1., resolution=sphere_resolution)
    edges = compute_edges_list(vertices, triangles)
    laplacian = compute_laplacian(vertices, edges)

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

    image_width = 96
    image_height = 96

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

    epochs_between_frames = 10
    epochs_between_previews = 100

    writer = imageio.get_writer(args.output_video, fps=20 / epochs_between_frames)
    sigma_val = 1e-4
    blur_radius = 0.1
    edge_loss_weight = 0.1
    laplacian_loss_weight = 0.1
    lr = 4.0
    momentum = 0.1
    optimizer = torch.optim.SGD([vertices], lr, momentum)
    def stepfn(e):
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

        loss = torch.mean((batched_renders[..., 3] - target_renders[..., 3])**2)
        loss += mesh_edge_loss(vertices, edges) * edge_loss_weight
        loss += mesh_laplacian_smoothing_loss(vertices, laplacian) * laplacian_loss_weight

        loss.backward()
        torch.nn.utils.clip_grad_norm_([vertices], 1.0)

        render = torch.reshape(batched_renders[0], [image_height, image_width, 4])
        if e % epochs_between_frames == 0:
            # write to video output
            frame = render.detach().numpy() # [image_height, image_width, 4]
            # black background
            frame = np.concatenate([
                frame[:,:,:3]*frame[:,:,3][:,:,None],
                np.ones([image_height, image_width, 1], dtype=np.float32)
            ], axis=-1)
            writer.append_data((255*frame).astype(np.uint8))

            print("\nappended frame {} to video output\n".format(e // epochs_between_frames))
        if e % epochs_between_previews == 0:
            # write a preview image to the preview directory
            preview_image_path = os.path.join(args.output_previews_dir, "preview_{:04d}.png".format(e))
            preview_obj_path = os.path.join(args.output_previews_dir, "preview_{:04d}.obj".format(e))
            result_image = render.detach().numpy()
            # Binarize the alpha channel to 0 or 1. In the raw output of the soft renderer,
            # it represents the probability that a triangle occupies the pixel. This will be
            # less than 1.0 for any pixel which is not entirely covered by a triangle, even if
            # the pixel is technically completely covered when considering all triangles. If we
            # don't binarize the value, we will get seams in the output along triangle edges.
            result_image[..., 3] = 1.0 * (result_image[..., 3] > 0.0)
            result_image = np.clip(result_image, 0., 1.).copy(order="C")
            io.imsave(preview_image_path, (result_image * 255.0).astype(np.uint8))

            obj_utils.save_obj(preview_obj_path, vertices, triangles)

            print("\nsaved previews to {} and {}\n".format(preview_image_path, preview_obj_path))

        return loss

    epochs = 1000
    loss_points = []
    for e in range(epochs):
        print("\nstep {} of {}\n".format(e, epochs))
        loss = optimizer.step(lambda: stepfn(e))
        loss_points.append(float(loss))

    writer.close()
    obj_utils.save_obj(args.output_model, vertices, triangles)

    x = np.arange(0, epochs, 1)
    y = np.array(loss_points)
    plt.plot(x, y)
    plt.show()

