"""Differentiable 3D rendering of a triangle mesh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import camera_utils
import rasterize_triangles


def mesh_renderer(
    vertices,
    triangles,
    normals,
    diffuse_colors,
    camera_position,
    camera_lookat,
    camera_up,
    light_positions,
    light_intensities,
    image_width,
    image_height,
    specular_colors=None,
    shininess_coefficients=None,
    ambient_color=None,
    fov_y=40.0,
    near_clip=0.01,
    far_clip=10.0):
    """Renders an input scene using phong shading, and returns an output image.

    Args:
      vertices: 3D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
      triangles: 2D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
      normals: 3D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
      diffuse_colors: 3D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0, 1] for
        each vertex.
      camera_position: 2D tensor withb shape [batch_size, 3] or 1D tensor with
        shape [3] specifying the XYZ world space camera position.
      camera_lookat: 2D tensor with shape [batch_size, 3] or 1D tensor with
        shape [3] containing an XYZ point along the center of the camera's gaze.
      camera_up: 2D tensor with shape [batch_size, 3] or 1D tensor with shape
        [3] containing the up direction for the camera. The camera will have
        no tilt with respect to this direction.
      light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
      light_intensities: a 3D tensor with shape [batch_size, light_count, 3].
        The RGB intensity values for each light. Intensities may be above 1.
      image_width: int specifying desired output image width in pixels.
      image_height: int specifying desired output image height in pixels.
      specular_colors: 3D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
        each vertex. If supplied, specular reflections will be computed, and
        both specular colors and shininess_coefficients are expected.
      shininess_coefficients: a 0D-2D float32 tensor with maximum shape
        [batch_size, vertex_count]. The phong shininess coefficient of each
        vertex. A 0D tensor or float gives a constant shininess coefficient of
        all vertices across all batches and images. A 1D tensor must have shape
        [batch_size], and a single shininess coefficient per image is used.
      ambient_color: a 2D tensor with shape [bath_size, 3]. The RGB ambient
        color, which is added to each pixel in the scene. If None, it is
        assumed to be black.
      fov_y: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        desired output image y field of view in degrees.
      near_clip: float, 0D tensor, or 1D tensor with shape [batch_size]
        specifying near clipping plane distance.
      far_clip: float, 0D tensor, or 1D tensor with shape [batch_size]
        specifying far clipping plane distance.

    Returns:
      A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
      containing the lit RGBA color values for each image at each pixel. RGB
      colors are the intensity values before tonemapping and can be in the range
      [0, infinity]. Clipping to the range [0, 1] with np.clip is likely
      reasonable for both viewing and training most scenes. More complex scenes
      with multiple lights should tone map color values for display only. One
      simple tonemapping approach is to rescale color values as x/(1+x); gamma
      compression is another common technique. Alpha values are zero for
      background pixels and near one for mesh pixels.
    Raises:
      ValueError: An invalid argument to the method is detected.
    """
    if len(vertices.shape) != 3 or vertices.shape[-1] != 3:
        raise ValueError(
            "Vertices must have shape [batch_size, vertex_count, 3].")
    batch_size = vertices.shape[0]
    if len(normals.shape) != 3 or normals.shape[-1] != 3:
        raise ValueError(
            "Normals must have shape [batch_size, vertex_count, 3].")
    if len(light_positions.shape) != 3 or light_positions.shape[-1] != 3:
        raise ValueError(
            "light_positions must have shape [batch_size, light_count, 3].")
    if len(light_intensities.shape) != 3 or light_intensities.shape[-1] != 3:
        raise ValueError(
            "light_intensities must have shape [batch_size, light_count, 3].")
    if len(diffuse_colors.shape) != 3 or diffuse_colors.shape[-1] != 3:
        raise ValueError(
            "diffuse_colors must have shape [batch_size, vertex_count, 3].")
    if (ambient_color is not None and
        list(ambient_color.shape) != [batch_size, 3]):
        raise ValueError("ambient_color must have shape [batch_size, 3].")
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
    if specular_colors is not None and shininess_coefficients is None:
        raise ValueError(
            "Specular colors were supplied without shininess coefficients.")
    if shininess_coefficients is not None and specular_colors is None:
        raise ValueError(
            "Shininess coefficients were supplied without specular colors.")
    if specular_colors is not None:
        # Since a 0D float32 tensor is accepted, also accept a float.
        if isinstance(shininess_coefficients, float):
            shininess_coefficients = torch.tensor(
                shininess_coefficients, dtype=torch.float32)
        if len(specular_colors.shape) != 3:
            raise ValueError("The specular colors must have shape [batch_size, "
                             "vertex_count, 3].")
        if len(shininess_coefficients.shape) > 2:
            raise ValueError("The shininess coefficients must have shape at "
                             "most [batch_size, vertex_count].")
        # If we don't have per-vertex coefficients, we can just reshape the
        # input shininess to broadcast later, rather than interpolating an
        # additional vertex attribute:
        if len(shininess_coefficients.shape) < 2:
            vertex_attributes = torch.cat(
                [normals, vertices, diffuse_colors, specular_colors], 2)
        else:
            vertex_attributes = torch.cat(
                [
                    normals, vertices, diffuse_colors, specular_colors,
                    torch.unsqueeze(shininess_coefficients, 2)
                ], 2)
    else:
        vertex_attributes = torch.cat([normals, vertices, diffuse_colors], 2)

    camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
                                           camera_up)

    perspective_transforms = camera_utils.perspective(
        image_width / image_height,
        fov_y,
        near_clip,
        far_clip)

    clip_space_transforms = torch.matmul(perspective_transforms, camera_matrices)

    pixel_attributes = rasterize_triangles.rasterize(
        vertices, vertex_attributes, triangles, triangles,
        clip_space_transforms, image_width, image_height,
        [-1] * vertex_attributes.shape[2])

    #################
