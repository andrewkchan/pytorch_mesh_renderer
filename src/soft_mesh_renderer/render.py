"""
Differentiable 3D rendering of a triangle mesh based on
the soft rasterization formulation from Liu 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from ..common import camera_utils
from .rasterize import rasterize

def compute_vertex_normals(vertices, triangles):
    """
    Computes vertex normals for a triangle mesh by first computing
    face normals, then averaging the normals on incident vertices.
    The resulting vectors are normalized.

    Args:
      vertices: 3D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
      triangles: 2D int32 tensor with shape [triangle_count, 3].

    Returns:
    - A tensor with shape [batch_size, vertex_count, 3] providing per-vertex normal
      vectors.
    """
    batch_size = vertices.shape[0]
    normals = torch.zeros_like(vertices)
    for b in range(batch_size):
        # vertices_faces[i][j] gives the vertex corresponding to triangles[i][j]
        vertices_faces = vertices[b, triangles.long(), :] # [vertex_count, 3, 3]
        normals[b].index_add_(0, triangles[:, 0].long(),
            torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                        vertices_faces[:, 2] - vertices_faces[:, 0])
        )
        normals[b].index_add_(0, triangles[:, 1].long(),
            torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                        vertices_faces[:, 0] - vertices_faces[:, 1])
        )
        normals[b].index_add_(0, triangles[:, 2].long(),
            torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                        vertices_faces[:, 1] - vertices_faces[:, 2])
        )
    normals = torch.nn.functional.normalize(normals, eps=1e-6, p=2, dim=-1)
    return normals

def render(
    vertices,
    triangles,
    diffuse_colors,
    camera_position,
    camera_lookat,
    camera_up,
    light_positions,
    light_intensities,
    image_width,
    image_height,
    sigma_val=1e-5,
    gamma_val=1e-4,
    fov_y=40.0,
    near_clip=0.01,
    far_clip=10.0):
    """Soft-renders an input scene using phong shading, and returns an output image.

    Args:
      vertices: 3D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
      triangles: 2D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a counter-clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
      diffuse_colors: 3D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0, 1] for
        each vertex.
      camera_position: 2D tensor with shape [batch_size, 3] or 1D tensor with
        shape [3] specifying the XYZ world space camera position.
      camera_lookat: 2D tensor with shape [batch_size, 3] or 1D tensor with
        shape [3] containing an XYZ point along the center of the camera's gaze.
      camera_up: 2D tensor with shape [batch_size, 3] or 1D tensor with shape
        [3] containing the up direction for the camera. The camera will have
        no tilt with respect to this direction.
      light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
        world space XYZ position of each light in the scene.
      light_intensities: a 3D tensor with shape [batch_size, light_count].
        The intensity values for each light. Intensities may be above 1.
      image_width: int specifying desired output image width in pixels.
      image_height: int specifying desired output image height in pixels.
      sigma_val: parameter controlling the sharpness of the coverage distribution
        for a single triangle. A smaller sigma leads to a sharper distribution.
      gamma_val: temperature parameter controlling uniformity of the triangle
        probability distribution for a pixel in the depth aggregation.
        When gamma is 0, all probability mass will fall into the triangle
        with highest z, matching the behavior of z-buffering.
      fov_y: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        desired output image y field of view in degrees.
      near_clip: float, 0D tensor, or 1D tensor with shape [batch_size]
        specifying near clipping plane distance.
      far_clip: float, 0D tensor, or 1D tensor with shape [batch_size]
        specifying far clipping plane distance.

    Returns:
        A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel.
        The RGB values are aggregated per-pixel according to the color aggregation
        formula in [1].
        The alpha values are aggregated per-pixel according to the silhouette
        formula in [1].

    [1] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    Raises:
      ValueError: An invalid argument to the method is detected.
    """
    if len(vertices.shape) != 3 or vertices.shape[-1] != 3:
        raise ValueError(
            "Vertices must have shape [batch_size, vertex_count, 3].")
    if len(triangles.shape) != 2 or triangles.shape[-1] != 3:
        raise ValueError(
            "Triangles must have shape [triangle_count, 3].")
    batch_size = vertices.shape[0]
    if len(light_positions.shape) != 3 or light_positions.shape[-1] != 3:
        raise ValueError(
            "light_positions must have shape [batch_size, light_count, 3].")
    if len(light_intensities.shape) != 2:
        raise ValueError(
            "light_intensities must have shape [batch_size, light_count].")
    if len(diffuse_colors.shape) != 3 or diffuse_colors.shape[-1] != 3:
        raise ValueError(
            "diffuse_colors must have shape [batch_size, vertex_count, 3].")
    if list(camera_position.shape) == [3]:
        camera_position = torch.unsqueeze(camera_position, 0).repeat(batch_size, 1)
    elif list(camera_position.shape) != [batch_size, 3]:
        raise ValueError(
            "camera_position must have shape [batch_size, 3] or [3].")
    if list(camera_lookat.shape) == [3]:
        camera_lookat = torch.unsqueeze(camera_lookat, 0).repeat(batch_size, 1)
    elif list(camera_lookat.shape) != [batch_size, 3]:
        raise ValueError(
            "camera_lookat must have shape [batch_size, 3] or [3].")
    if list(camera_up.shape) == [3]:
        camera_up = torch.unsqueeze(camera_up, 0).repeat(batch_size, 1)
    elif list(camera_up.shape) != [batch_size, 3]:
        raise ValueError("camera_up must have shape [batch_size, 3] or [3].")
    if isinstance(fov_y, float):
        fov_y = torch.tensor(batch_size * [fov_y], dtype=torch.float32)
    elif len(fov_y.shape) == 0:
        fov_y = torch.unsqueeze(fov_y, 0).repeat(batch_size)
    elif list(fov_y.shape) != [batch_size]:
        raise ValueError("fov_y must be a float, a 0D tensor, or a 1D tensor "
                         "with shape [batch_size].")
    if isinstance(near_clip, float):
        near_clip = torch.tensor(batch_size * [near_clip], dtype=torch.float32)
    elif len(near_clip.shape) == 0:
        near_clip = torch.unsqueeze(near_clip, 0).repeat(batch_size)
    elif list(near_clip.shape) != [batch_size]:
        raise ValueError("near_clip must be a float, a 0D tensor, or a 1D "
                         "tensor with shape [batch_size].")
    if isinstance(far_clip, float):
        far_clip = torch.tensor(batch_size * [far_clip], dtype=torch.float32)
    elif len(far_clip.shape) == 0:
        far_clip = torch.unsqueeze(far_clip, 0).repeat(batch_size)
    elif list(far_clip.shape) != [batch_size]:
        raise ValueError("far_clip must be a float, a 0D tensor, or a 1D "
                         "tensor with shape [batch_size].")

    camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
                                           camera_up)

    perspective_transforms = camera_utils.perspective(
        image_width / image_height,
        fov_y,
        near_clip,
        far_clip)

    clip_space_transforms = torch.matmul(perspective_transforms, camera_matrices)
    normals = compute_vertex_normals(vertices, triangles)

    return rasterize(
        vertices,
        triangles,
        ### vertex attributes
        normals,
        diffuse_colors,
        ### lighting
        light_positions,
        light_intensities,
        ###
        clip_space_transforms,
        image_width,
        image_height,
        sigma_val,
        gamma_val
    )