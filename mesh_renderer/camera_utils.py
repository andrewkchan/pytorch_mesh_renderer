from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
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
