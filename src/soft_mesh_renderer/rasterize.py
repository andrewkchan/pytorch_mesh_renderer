"""
Differentiable triangle rasterizer using soft rasterization formulation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from ..common import camera_utils

def rasterize(
    world_space_vertices,
    triangles,
    ### vertex attributes
    normals,
    diffuse_colors,
    ### lighting
    light_positions,
    light_intensities,
    ###
    camera_matrices,
    image_width,
    image_height,
    sigma_val,
    gamma_val
):
    """
    Soft-rasterize a mesh, interpolating vertex attributes, lighting with phong shading,
    and soft-aggregating the result for every pixel.

    Args:
        world_space_vertices: 3D float32 tensor of xyz positions with shape
            [batch_size, vertex_count, 3].
        triangles: 2D int32 tensor with shape [triangle_count, 3]. Each triplet
            should contain vertex indices describing a triangle such that the
            triangle's normal points toward the viewer if the forward order of
            the triplet defines a counter-clockwise winding of the vertices. Gradients
            with respect to this tensor are not available.

        normals: 3D float32 tensor with shape [batch_size, vertex_count, 3]. Each
            triplet is the xyz vertex normal for its corresponding vertex. Each
            vector is assumed to be already normalized.
        diffuse_colors: 3D float32 tensor with shape [batch_size,
            vertex_count, 3]. The RGB diffuse reflection in the range [0, 1] for
            each vertex.

        light_positions: a 3D tensor with shape [batch_size, light_count, 3]. The
            world space XYZ position of each light in the scene.
        light_intensities: a 3D tensor with shape [batch_size, light_count].
            The intensity values for each light. Intensities may be above 1.

        camera_matrices: 3D float tensor with shape [batch_size, 4, 4] containing
            model-view-perspective projection matrices.
        image_width: int specifying desired output image width in pixels.
        image_height: int specifying desired output image height in pixels.
        sigma_val: parameter controlling the sharpness of the coverage distribution
            for a single triangle. A smaller sigma leads to a sharper distribution.
        gamma_val: temperature parameter controlling uniformity of the triangle
            probability distribution for a pixel in the depth aggregation.
            When gamma is 0, all probability mass will fall into the triangle
            with highest z, matching the behavior of z-buffering.

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
    vertex_count = world_space_vertices.shape[1]
    batch_size = world_space_vertices.shape[0]

    clip_space_vertices = camera_utils.transform_homogeneous(
        camera_matrices, world_space_vertices)

    batch_images = []

    for b in range(batch_size):
        image = rasterize_batch(
            clip_space_vertices[b, :, :],
            triangles,
            ### vertex attributes
            world_space_vertices[b, :, :],
            normals[b, :, :],
            diffuse_colors[b, :, :],
            ### lighting
            light_positions[b, :, :],
            light_intensities[b, :],
            ###
            image_width,
            image_height,
            sigma_val,
            gamma_val
        )
        batch_images.append(image)

    return torch.stack(batch_images, 0)

# Returns the signed area of the parallelogram
# with edges v0p and v01. All inputs should be tensors
# of shape [2] or [3].
#
# The area is positive if point p is on the right side
# of the segment going from v0 to v1 (so that [p, v0, v1]
# winds clockwise) and negative if p is on the left (so
# that [p, v0, v1] winds counter-clockwise).
def edge_function(p, v0, v1):
    v0p = p - v0
    v01 = v1 - v0
    return v0p[0] * v01[1] - v0p[1] * v01[0]

# Returns barycentric coordinates of a 3D point P w.r.t. triangle v0, v1, v2.
# The input `M_inv` should be the inverse of a 3x3 matrix where the columns are the vertices.
def barycentric(M_inv, p):
    return M_inv @ p

# Returns barycentric coordinates of a point P (in homogeneous 3D coordinates xyz)
# w.r.t. triangle v0, v1, v2, the same for the point on the edge of the triangle nearest to P,
# and the distance between them.
# Args:
# - p: 3D point, a tensor with shape [3].
# - M: A 3x3 matrix where the columns are the vertices v0, v1, v2 of the triangle.
# - M_inv: The inverse of M.
#
# Returns:
# - bc_p: 1D tensor of shape [3] giving barycentric coordinates for p.
#         If p is outside the triangle, one of the coordinates will be negative.
# - mindist_sq: scalar tensor (float) giving the squared distance from p to the nearest point.
# - bc_edge: 1D tensor of shape [3] giving barycentric coordinates for the nearest point
#               on the edge of the triangle.
def barycentric_edge(M, M_inv, p):
    bc_p = barycentric(M_inv, p)
    v01_nearest, t01 = point_to_segment_nearest(p[:2], M[:, 0][:2], M[:, 1][:2])
    v12_nearest, t12 = point_to_segment_nearest(p[:2], M[:, 1][:2], M[:, 2][:2])
    v20_nearest, t20 = point_to_segment_nearest(p[:2], M[:, 2][:2], M[:, 0][:2])
    d = torch.stack([v01_nearest, v12_nearest, v20_nearest]) - p[:2]
    mindist_sq, argmin = torch.min(torch.sum(d * d, dim=-1), dim=0)
    if argmin == 0:
        return bc_p, mindist_sq, torch.stack([1. - t01, t01, torch.tensor(0.)])
    elif argmin == 1:
        return bc_p, mindist_sq, torch.stack([torch.tensor(0.), 1. - t12, t12])
    else:
        return bc_p, mindist_sq, torch.stack([t20, torch.tensor(0.), 1. - t20])

# Returns the point on a 2D line segment which is nearest to the input point,
# and the number t between [0, 1] giving how far that is on the segment.
#
# Args:
# - p: 2D point, a tensor with shape [2] that we want to project on the line segment.
# - a: 2D point, a tensor with shape [2]. Start of the line segment.
# - b: 2D point, a tensor with shape [2]. End of the line segment.
#
# Returns:
# - x: 2D point, the point on the line segment nearest p.
# - t: Number between [0, 1] giving the normalized distance from `a` to `x`.
def point_to_segment_nearest(p, a, b):
    ab = b - a
    len_ab = torch.linalg.vector_norm(b - a, ord=2)
    n = ab / max(len_ab, 1e-12)
    proj_p_n = torch.dot(p - a, n) * n
    t = torch.clamp(torch.dot(proj_p_n, n) / len_ab, 0., 1.)
    x = a + t * ab
    return x, t

# Samples the diffuse texture of the triangle at the given barycentric
# coordinates, then returns the corresponding RGBA color with phong shading
# applied to it.
# Returns:
# - a tensor of shape [3] giving the lit RGB value for this pixel
def compute_shaded_color(
    bc,
    triangle,
    ### vertex attributes
    world_space_vertices,
    normals,
    diffuse_colors,
    ### lighting
    light_positions,
    light_intensities,
):
    light_count = len(light_positions)
    diffuse_color = bc @ diffuse_colors[triangle, :] # [3]
    p = bc @ world_space_vertices[triangle, :] # [3]
    n = torch.nn.functional.normalize(bc @ normals[triangle, :], p=2, dim=-1) # [3]
    dirs_to_lights = torch.nn.functional.normalize(
        light_positions - p, p=2, dim=-1) # [light_count, 3]

    # Surfaces should only be illuminated when the light and normal face
    # one another (e.g. dot product is non-negative)
    normals_dot_lights = torch.clamp(
        torch.sum(dirs_to_lights * n, dim=-1),
        0.0, 1.0) # [light_count]
    diffuse_output = diffuse_color * torch.sum(normals_dot_lights * light_intensities, dim=-1) # [3]

    return diffuse_output

SHOW_DEBUG_LOGS = False
EPS = 1e-10 # used to give background color a constant small probability
def rasterize_batch(
    clip_space_vertices,
    triangles,
    ### vertex attributes
    world_space_vertices,
    normals,
    diffuse_colors,
    ### lighting
    light_positions,
    light_intensities,
    ###
    image_width,
    image_height,
    sigma_val,
    gamma_val,
    blur_radius=0.01
):
    """
    Soft-rasterize a mesh already transformed to clip space.
    Non-batched function.

    Args:
        clip_space_vertices: 2D float32 tensor of homogeneous vertices (xyzw)
            with shape [vertex_count, 4].
        triangles: 2D int32 tensor with shape [triangle_count, 3]. Each triplet
            should contain vertex indices describing a triangle such that the
            triangle's normal points toward the viewer if the forward order of
            the triplet defines a counter-clockwise winding of the vertices. Gradients
            with respect to this tensor are not available.

        world_space_vertices: 2D float32 tensor of xyz positions with shape
            [vertex_count, 3].
        normals: 2D float32 tensor with shape [vertex_count, 3]. Each
            triplet is the xyz vertex normal for its corresponding vertex. Each
            vector is assumed to be already normalized.
        diffuse_colors: 2D float32 tensor with shape [vertex_count, 3]. The RGB
            diffuse reflection in the range [0, 1] for each vertex.

        light_positions: a 2D tensor with shape [light_count, 3]. The world space
            XYZ position of each light in the scene.
        light_intensities: a 1D tensor with shape [light_count].
            The intensity values for each light. Intensities may be above 1.

        image_width: int specifying desired output image width in pixels.
        image_height: int specifying desired output image height in pixels.
        sigma_val: parameter controlling the sharpness of the coverage distribution
            for a single triangle. A smaller sigma leads to a sharper distribution.
        gamma_val: temperature parameter controlling uniformity of the triangle
            probability distribution for a pixel in the depth aggregation.
            When gamma is 0, all probability mass will fall into the triangle
            with highest z, matching the behavior of z-buffering.
        blur_radius: float specifying the cutoff radius of soft-rasterization sampling
            in NDC-space.

    Returns:
        A 3D float32 tensor of shape [image_height, image_width, 4]
        containing the lit RGBA color values at each pixel.
        The RGB values are aggregated per-pixel according to the color aggregation
        formula in [1].
        The alpha values are aggregated per-pixel according to the silhouette
        formula in [1].

    [1] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    sq_blur_radius = blur_radius**2
    result = torch.zeros([image_height, image_width, 4], dtype=torch.float32)

    ndc_face_matrices = torch.zeros([len(triangles), 3, 3], dtype=torch.float32)
    ndc_2d_face_matrices_inv = torch.zeros([len(triangles), 3, 3], dtype=torch.float32)
    ndc_face_areas = torch.zeros([len(triangles)], dtype=torch.float32)
    for i in range(len(triangles)):
        triangle = triangles[i]
        clip_v012 = clip_space_vertices[triangle] # shape: [3, 4]
        clip_v012_w = clip_v012[:, [3]] # shape: [3, 1]

        ndc_M = (clip_v012 / (clip_v012_w)).T[:3, :] # [3, 3], each column is a vertex
        ndc_face_matrices[i, :, :] = ndc_M

        ndc_2d_M = ndc_M.clone()
        ndc_2d_M[2, :] = torch.tensor([1., 1., 1.])
        try:
            ndc_2d_M_inv = ndc_2d_M.inverse()
        except Exception:
            # NDC-space vertex basis is not invertible, meaning triangle is
            # degenerate when projected (zero area).
            continue
        ndc_2d_face_matrices_inv[i, :, :] = ndc_2d_M_inv
        ndc_face_areas[i] = edge_function(ndc_M[:, 0], ndc_M[:, 1], ndc_M[:, 2])

    total_samples = 0
    for y in range(image_height):

        row_samples_drawn = 0
        row_max_samples_drawn = 0

        for x in range(image_width):
            ndc_x = 2.0 * ((x + 0.5) / image_width) - 1.0
            ndc_y = -2.0 * ((y + 0.5) / image_height) + 1.0 # invert y
            ndc_p = torch.tensor([ndc_x, ndc_y, 1.0])

            soft_weights = torch.zeros([len(triangles)])
            soft_fragments = torch.zeros([len(triangles)])
            soft_colors = torch.zeros([len(triangles), 3])

            samples_drawn = 0
            for i in range(len(triangles)):
                triangle = triangles[i]

                clip_v012 = clip_space_vertices[triangle] # shape: [3, 4]
                clip_v012_w = clip_v012[:, [3]] # shape: [3, 1]
                ndc_M = ndc_face_matrices[i] # [3, 3]
                ndc_depths = ndc_M.T[:, [2]] # [3, 1]
                if ndc_face_areas[i] > 0:
                    # Back-face culling: skip triangles facing away from the camera.
                    continue
                elif ndc_face_areas[i] == 0:
                    # Skip degenerate triangles with zero area.
                    continue
                ndc_2d_M_inv = ndc_2d_face_matrices_inv[i]

                # fast distance culling: check if pixel is outside the
                # triangle's bounding box inflated by blur_radius
                if (ndc_x < torch.min(ndc_M[0, :]) - blur_radius or
                    ndc_x > torch.max(ndc_M[0, :]) + blur_radius or
                    ndc_y < torch.min(ndc_M[1, :]) - blur_radius or
                    ndc_y > torch.max(ndc_M[1, :]) + blur_radius):
                    continue
                bc_screen, sq_dist, bc_edge_screen = barycentric_edge(
                    # Note: ndc_2d_M_inv is the inverse of `ndc_M` with uniform z-components,
                    # not `ndc_M` itself. This is ok because we only use the `M` matrix in
                    # this function to extract the x and y components of face vertices.
                    ndc_M,
                    ndc_2d_M_inv,
                    ndc_p
                )
                is_inside = not torch.any(bc_screen < 0.)

                # slow distance culling: check if pixel is too far from sample point
                if not is_inside and sq_dist > sq_blur_radius:
                    continue

                # Get perspective-correct barycentric coordinates for the point to sample from
                # by un-doing the perspective projection on the screen-space barycentrics.
                sample_bc = torch.nn.functional.normalize(
                    # If p is inside the triangle, sample from p itself.
                    # Otherwise, sample from the point inside the triangle nearest to p.
                    (bc_screen if is_inside else bc_edge_screen)
                    / clip_v012_w.T[0],
                    dim=0, p=1
                ) # [3]

                # Get normalized depth of nearest points in NDC-space.
                z = sample_bc @ ndc_depths # Range [-1, +1] where -1 is near plane
                # Map to range (0, 1) where 1.0 is near plane, 0.0 is far plane
                z = 0.5 - z/2.

                if z < 0.0 or z > 1.0:
                    # Sample point is out of screen, pass
                    continue

                soft_colors[i, :3] = compute_shaded_color(
                    sample_bc,
                    triangle,
                    ### vertex attributes
                    world_space_vertices,
                    normals,
                    diffuse_colors,
                    ### lighting
                    light_positions,
                    light_intensities,
                )

                sgn = 1. if is_inside else -1.
                soft_fragments[i] = torch.special.expit(sgn * sq_dist / sigma_val)

                # Set these equal to the un-exponentiated logits.
                # We shouldn't exponentiate until we can adjust the maximum value
                # below to avoid overflow.
                soft_weights[i] = z / gamma_val
                samples_drawn += 1

            max_soft_weight = max(torch.max(soft_weights), torch.tensor(EPS / gamma_val))
            soft_weights = soft_fragments * torch.exp(soft_weights - max_soft_weight)

            # background weight should never be zero.
            bg_weight = max(torch.exp(EPS / gamma_val - max_soft_weight), EPS)

            # normalize all logits
            sum_weights = torch.sum(soft_weights) + bg_weight
            soft_weights = soft_weights / sum_weights

            # bg color is transparent, otherwise we'd add `(bg_weight / sum_weights) * bg_color`
            result[y][x][:3] = soft_weights @ soft_colors

            # Compute the silhouette score, which is based on the probability that
            # at least 1 triangle covers the pixel. This is 1 - probability that
            # all triangles do not cover the pixel.
            silhouette = 1.0 - torch.prod((1.0 - soft_fragments))
            result[y][x][3] = silhouette

            row_samples_drawn += samples_drawn
            row_max_samples_drawn = max(row_max_samples_drawn, samples_drawn)
            total_samples += samples_drawn
        if SHOW_DEBUG_LOGS:
            print("drew {} samples (max={}) for row y={}".format(row_samples_drawn, row_max_samples_drawn, y))
    if SHOW_DEBUG_LOGS:
        print("drew {} samples total".format(total_samples))

    return result