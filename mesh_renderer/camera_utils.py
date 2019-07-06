from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch


def euler_matrices(angles):
    """Compute a XYZ Tait-Bryan (improper Euler angle) rotation.

    Return 4x4 matrices for convenient multiplication with other transformations.

    Args:
      angles: a [batch_size, 3] tensor containing X, Y, and Z angles in radians.

    Returns:
      a [batch_size, 4, 4] tensor of matrices.
    """
    s = torch.sin(angles)
    c = torch.cos(angles)
    # Rename variables for readability in the matrix definition below.
    c0, c1, c2 = (c[:, 0], c[:, 1], c[:, 2])
    s0, s1, s2 = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0])
    ones = torch.ones_like(s[:, 0])

    flattened = torch.cat(
        [
            c2*c1, c2*s1*s0 - c0*s2, s2*s0 + c2*c0*s1, zeros,
            c1*s2, c2*c0 + s2*s1*s0, c0*s2*s1 - c2*s0, zeros,
            -s1, c1*s0, c1*c0, zeros,
            zeros, zeros, zeros, ones
        ],
        dim=0)
    reshaped = torch.reshape(flattened, [4, 4, -1])
    return torch.transpose(reshaped, [2, 0, 1])


def look_at(eye, center, world_up):
    """Compute camera viewing matrices.

    Functionality mimes gluLookAt (external/GL/glu/include/GLU/glu.h).

    Args:
        eye: 2D float32 tensor with shape [batch_size, 3] containing the XYZ
            world space position of the camera.
        center: 2D float32 tensor with shape [batch_size, 3] containing a
            position along the center of the camera's gaze line.
        world_up: 2D float32 tensor with shape [batch_size, 3] specifying the
            world's up direction; the output camera will have no tilt with
            respect to this direction.

    Returns:
        A [batch_size, 4, 4] float tensor containing a right-handed camera
        extrinsics matrix that maps points from world space to points in eye
        space.
    """
    batch_size = center.shape[0]
    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = torch.tensor(
        np.linalg.norm(forward, ord=None, axis=1, keepdims=True))
    np.testing.assert_array_less(vector_degeneracy_cutoff, forward_norm,
        message="Camera matrix is degenerate because eye and center are close.")
    forward = forward/forward_norm

    to_side = torch.cross(forward, world_up)
    to_side_norm = torch.tensor(
        np.linalg.norm(to_side, ord=None, axis=1, keepdims=True))
    np.testing.assert_array_less(vector_degeneracy_cutoff, to_side_norm,
        message="Camera matrix is degenerate because up and gaze are too close "
                "or because up is degenerate.")
    to_side = to_side/to_side_norm
    cam_up = torch.cross(to_side, forward)

    w_column = torch.tensor(
        batch_size * [[0., 0., 0., 1.]], dtype=torch.float32) # [batch_size, 4]
    w_column = torch.reshape(w_column, [batch_size, 4, 1])
    view_rotation = torch.stack(
        [to_side, cam_up, -forward,
         torch.zeros_like(to_side, dtype=torch.float32)],
        axis=1) # [batch_size, 4, 3] matrix
    view_rotation = torch.cat([view_rotation, w_column],
                              axis=2) # [batch_size, 4, 4]

    identity_batch = torch.unsqueeze(torch.eye(3), 0).repeat([batch_size, 1, 1])
    view_translation = torch.cat([identity_batch, torch.unsqueeze(-eye, 2)], 2)
    view_translation = torch.cat(
        [view_translation,
         torch.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = torch.matmul(view_rotation, view_translation)
    return camera_matrices


def perspective(aspect_ratio, fov_y, near_clip, far_clip):
    """Computes perspective transformation matrices.

    Functionality mimes gluPerspective (external/GL/glu/include/GLU/glu.h).
    See:
    https://unspecified.wordpress.com/2012/06/21/calculating-the-gluperspective-matrix-and-other-opengl-matrix-maths/

    Args:
        aspect_ratio: float value specifying the image aspect ratio
            (width/height).
        fov_y: 1D float32 Tensor with shape [batch_size] specifying output
            vertical field of views in degrees.
        near_clip: 1D float32 Tensor with shape [batch_size] specifying near
            clipping plane distance.
        far_clip: 1D float32 Tensor with shape [batch_size] specifying far
            clipping plane distance.

    Returns:
        A [batch_size, 4, 4] float tensor that maps from right-handed points in
        eye space to left-handed points in clip space.
    """
    # The multiplication of fov_y by pi/360.0 simultaneously converts to radians
    # and adds the half-angle factor of .5.
    focal_lengths_y = 1.0 / torch.tan(fov_y * (math.pi / 360.0))
    depth_range = far_clip - near_clip
    p_22 = -(far_clip + near_clip) / depth_range
    p_23 = -2.0 * (far_clip * near_clip / depth_range)

    zeros = torch.zeros_like(p_23, dtype=torch.float32)
    perspective_transform = torch.cat(
        [
            focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
            zeros, focal_lengths_y, zeros, zeros,
            zeros, zeros, p_22, p_23,
            zeros, zeros, -torch.ones_like(p_23, dtype=torch.float32), zeros
        ], axis=0)
    perspective_transform = torch.reshape(perspective_transform, [4, 4, -1])
    return torch.transpose(perspective_transform, [2, 0, 1])
