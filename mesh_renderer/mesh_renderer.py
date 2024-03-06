"""Differentiable 3D rendering of a triangle mesh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from mesh_renderer import camera_utils
from mesh_renderer.rasterize_triangles import rasterize


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
      camera_position: 2D tensor with shape [batch_size, 3] or 1D tensor with
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
      ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
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

    pixel_attributes = rasterize(
        vertices, vertex_attributes, triangles,
        clip_space_transforms, image_width, image_height,
        torch.tensor([-1] * vertex_attributes.shape[2]))

    # Extract the interpolated vertex attributes from the pixel buffer and
    # supply them to the shader:
    pixel_normals = torch.nn.functional.normalize(
        pixel_attributes[:, :, :, 0:3], p=2, dim=3)
    pixel_positions = pixel_attributes[:, :, :, 3:6]
    diffuse_colors = pixel_attributes[:, :, :, 6:9]
    if specular_colors is not None:
        specular_colors = pixel_attributes[:, :, :, 9:12]
        # Retrieve the interpolated shininess coefficients if necessary, or just
        # reshape our input for broadcasting:
        if len(shininess_coefficients.shape) == 2:
            shininess_coefficients = pixel_attributes[:, :, :, 12]
        else:
            shininess_coefficients = torch.reshape(
                shininess_coefficients, [-1, 1, 1])

    pixel_mask = (diffuse_colors >= 0.0).any(dim=3).type(torch.float32)

    renders = phong_shader(
        normals=pixel_normals,
        alphas=pixel_mask,
        pixel_positions=pixel_positions,
        light_positions=light_positions,
        light_intensities=light_intensities,
        diffuse_colors=diffuse_colors,
        camera_position=camera_position if specular_colors is not None else None,
        specular_colors=specular_colors,
        shininess_coefficients=shininess_coefficients,
        ambient_color=ambient_color)
    return renders


def phong_shader(normals,
                 alphas,
                 pixel_positions,
                 light_positions,
                 light_intensities,
                 diffuse_colors=None,
                 camera_position=None,
                 specular_colors=None,
                 shininess_coefficients=None,
                 ambient_color=None):
    """Compute pixelwise lighting from rasterized buffers with the Phong model.

    Args:
        normals: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the world space XYZ normal
            for the corresponding pixel. Should be already normalized.
        alphas: a 3D float32 tensor with shape [batch_size, image_height,
            image_width]. The inner dimension is the alpha value (transparency)
            for the corresponding pixel.
        pixel_positions: a 4D float32 tensor with shape [batch_size,
            image_height, image_width, 3]. The inner dimension is the world
            space XYZ position for the corresponding pixel.
        light_positions: a 3D tensor with shape [batch_size, light_count, 3].
            The XYZ position of each light in the scene. In the same coordinate
            space as pixel_positions.
        light_intensities: a 3D tensor with shape [batch_size, light_count, 3].
            The RGB intensity values for each light. Intensities may be above 1.
        diffuse_colors: a 4D float32 tensor with shape [batch_size, image_height,
            image_width, 3]. The inner dimension is the diffuse RGB coefficients
            at a pixel in the range [0, 1].
        camera_position: a 1D tensor with shape [batch_size, 3]. The XYZ camera
            position in the scene. If supplied, specular reflections will be
            computed. If not supplied, specular_colors and shininess_coefficients
            are expected to be None. In the same coordinate space as
            pixel_positions.
        specular_colors: a 4D float32 tensor with shape [batch_size,
            image_height, image_width, 3]. The inner dimension is the specular
            RGB coefficients at a pixel in the range [0, 1]. If None, assumed
            to be torch.zeros().
        shininess_coefficients: a 3D float32 tensor that is broadcasted to
            shape [batch_size, image_height, image_width]. The inner dimension
            is the shininess coefficient for the object at a pixel. Dimensions
            that are constant can be given length 1, so [batch_size, 1, 1] and
            [1, 1, 1] are also valid input shapes.
        ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
            color, which is added to each pixel before tone mapping. If None,
            it is assumed to be torch.zeros().

    Returns:
        A 4D float32 tensor of shape [batch_size, image_height, image_width, 4]
        containing the lit RGBA color values for each image at each pixel.
        Colors are in the range [0, 1].

    Raises:
        ValueError: An invalid argument to the method is detected.
    """
    batch_size, image_height, image_width = [s for s in normals.shape[:-1]]
    light_count = light_positions.shape[1]
    pixel_count = image_height * image_width
    # Reshape all values to easily do pixelwise computations:
    normals = torch.reshape(normals, [batch_size, -1, 3])
    alphas = torch.reshape(alphas, [batch_size, -1, 1])
    diffuse_colors = torch.reshape(diffuse_colors, [batch_size, -1, 3])
    if camera_position is not None:
        specular_colors = torch.reshape(specular_colors, [batch_size, -1, 3])

    # Ambient component
    output_colors = torch.zeros([batch_size, image_height * image_width, 3])
    if ambient_color is not None:
        ambient_reshaped = torch.unsqueeze(ambient_color, 1)
        output_colors = output_colors + ambient_reshaped * diffuse_colors

    # Diffuse component
    pixel_positions = torch.reshape(pixel_positions, [batch_size, -1, 3])
    per_light_pixel_positions = torch.stack(
        [pixel_positions] * light_count,
        dim=1) # [batch_size, light_count, pixel_count, 3]
    directions_to_lights = torch.nn.functional.normalize(
        torch.unsqueeze(light_positions, 2) - per_light_pixel_positions,
        p=2,
        dim=3) # [batch_size, light_count, pixel_count, 3]
    # The specular component should only contribute when the light and normal
    # face one another (i.e. the dot product is nonnegative):
    normals_dot_lights = torch.clamp(
        torch.sum(
            torch.unsqueeze(normals, 1) * directions_to_lights, dim=3),
        0.0, 1.0) # [batch_size, light_count, pixel_count]
    diffuse_output = (
        torch.unsqueeze(diffuse_colors, 1) *
        torch.unsqueeze(normals_dot_lights, 3) *
        torch.unsqueeze(light_intensities, 2))
    diffuse_output = torch.sum(diffuse_output, dim=1) # [batch_size, pixel_count, 3]
    output_colors = output_colors + diffuse_output

    # Specular component
    if camera_position is not None:
        camera_position = torch.reshape(camera_position, [batch_size, 1, 3])
        mirror_reflection_direction = torch.nn.functional.normalize(
            2.0 * torch.unsqueeze(normals_dot_lights, 3) * torch.unsqueeze(
                normals, 1) - directions_to_lights,
            p=2,
            dim=3) # [batch_size, light_count, pixel_count, 3]
        direction_to_camera = torch.nn.functional.normalize(
            camera_position - pixel_positions,
            p=2,
            dim=2) # [batch_size, pixel_count, 3]
        reflection_direction_dot_camera_direction = torch.sum(
            mirror_reflection_direction * torch.unsqueeze(direction_to_camera, 1),
            dim=3)
        # The specular component should only contribute when the reflection is
        # external:
        reflection_direction_dot_camera_direction = torch.clamp(
            torch.nn.functional.normalize(
                reflection_direction_dot_camera_direction,
                p=2,
                dim=2),
            0.0,
            1.0)
        # The specular component should also only contribute when the diffuse
        # component contributes:
        reflection_direction_dot_camera_direction = torch.where(
            normals_dot_lights != 0.0,
            reflection_direction_dot_camera_direction,
            torch.zeros_like(
                reflection_direction_dot_camera_direction,
                dtype=torch.float32))
        # Reshape to support broadcasting the shininess coefficient, which
        # rarely varies per-vertex:
        reflection_direction_dot_camera_direction = torch.reshape(
            reflection_direction_dot_camera_direction,
            [batch_size, light_count, image_height, image_width])
        shininess_coefficients = torch.unsqueeze(shininess_coefficients, 1)
        specularity = torch.reshape(
            torch.pow(reflection_direction_dot_camera_direction,
                      shininess_coefficients),
            [batch_size, light_count, pixel_count, 1])
        specular_output = (
            torch.unsqueeze(specular_colors, 1) * specularity *
            torch.unsqueeze(light_intensities, 2)
        )
        specular_output = torch.sum(specular_output, dim=1)
        output_colors = output_colors + specular_output
    rgb_images = torch.reshape(
        output_colors,
        [batch_size, image_height, image_width, 3])
    alpha_images = torch.reshape(
        alphas,
        [batch_size, image_height, image_width, 1])
    valid_rgb_values = torch.cat(3 * [alpha_images > 0.5], dim=3)
    rgb_images = torch.where(
        valid_rgb_values,
        rgb_images,
        torch.zeros_like(rgb_images, dtype=torch.float32))
    return torch.flip(
        torch.cat([rgb_images, alpha_images], dim=3),
        dims=[1])


def tone_mapper(image, gamma):
    """Apply gamma correction to the input image.

    Tone maps the input image batch in order to make scenes with a high dynamic
    range viewable. The gamma correction factor is computed separately per
    image, but is shared between all provided channels. The exact function
    computed is:

    image_out = A*image_in^gamma, where A is an image-wide constant computed
    so that the maximum image value is approximately 1. The correction is
    applied to all channels.

    Args:
        image: 4D float32 tensor with shape [batch_size, image_height,
            image_width, channel_count]. The batch of images to tone map.
        gamma: 0D float32 nonnegative tensor. Values of gamma below 1 compress
            relative contrast in the image, and values above one increase it.
            A value of 1 is equivalent to scaling the image to have a max value
            of 1.
    Returns:
        4D float32 tensor with shape [batch_size, image_height, image_width,
        channel_count]. Contains the gamma-corrected images, clipped to the
        range [0, 1].
    """
    batch_size = image.shape[0]
    corrected_image = torch.pow(image, gamma)
    image_max = torch.max(
        torch.reshape(corrected_image, [batch_size, -1]), 1).values
    scaled_image = (
        corrected_image / torch.reshape(image_max, [batch_size, 1, 1, 1]))
    return torch.clamp(scaled_image, 0.0, 1.0)
