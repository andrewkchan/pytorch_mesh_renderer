from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from skimage import io
import torch
from itertools import product


def check_jacobians_are_nearly_equal(theoretical,
                                     numerical,
                                     outlier_relative_error_threshold,
                                     max_outlier_fraction,
                                     include_jacobians_in_error_message=False):
    """Compare two Jacobian matrices, allowing for some fraction of outliers.

    Args:
    theoretical: 2D numpy array containing a Jacobian matrix with entries
        computed via gradient functions. The layout should be as in the output
        of torch.autograd.gradcheck.get_analytical_jacobian.
    numerical: 2D numpy array of the same shape as theoretical containing a
        Jacobian matrix with entries computed via finite difference
        approximations. The layout should be as in the output of
        torch.autograd.gradcheck.get_numerical_jacobian.
    outlier_relative_error_threshold: float prescribing the max relative
        error (from the finite difference approximation) tolerated before an
        entry is considered an outlier.
    max_outlier_fraction: float defining the max fraction of entries in
        theoreticdal that may be outliers before the check returns False.
    include_jacobians_in_error_message: bool defining whether the jacobian
        matrices should be included in the return message if the test fails.

    Returns:
    A tuple (success: bool, error_msg: str).
    """
    outlier_gradients = np.abs(
        numerical - theoretical) / numerical > outlier_relative_error_threshold
    outlier_fraction = (
        np.count_nonzero(outlier_gradients) / np.prod(numerical.shape[:2]))
    jacobians_match = outlier_fraction <= max_outlier_fraction

    message = (
        " %f of theoretical gradients are relative outliers, but the maximum"
        "allowable fraction is %f " % (outlier_fraction, max_outlier_fraction))
    if include_jacobians_in_error_message:
        # The gradient checker convention is the typical Jacobian transposed:
        message += ("\nNumerical Jacobian:\n%s\nTheoretical Jacobian:\n%s" %
                    (repr(numerical.T), repr(theoretical.T)))
    return jacobians_match, message


def get_analytical_jacobian(input, output):
    """Compute the analytical jacobian for a function with a single
       differentiable argument.
    """
    jacobian = torch.zeros(input.numel(), output.numel())
    grad_output = torch.zeros_like(output)
    flag_grad_output = grad_output.view(-1)

    for i in range(flag_grad_output.numel()):
        flag_grad_output.zero_()
        flag_grad_output[i] = 1
        d_x = torch.autograd.grad(output, [input], grad_output,
                                  retain_graph=True, allow_unused=True)[0]
        x = input
        if jacobian.numel() != 0:
            if d_x is None:
                jacobian[:, i].zero_()
            else:
                d_x_dense = (d_x.to_dense()
                             if not d_x.layout == torch.strided else d_x)
                assert jacobian[:, i].numel() == d_x_dense.numel()
                jacobian[:, i] = d_x_dense.contiguous().view(-1)

    return jacobian


def get_numerical_jacobian(fn, input, eps=1e-3):
    """Compute the numerical Jacobian using finite differences.

    Args:
        fn: The function to differentiate.
        input: input to `fn`
        eps: Finite difference epsilon.
    """
    output_size = fn(input).numel()
    jacobian = torch.zeros(input.numel(), output_size)
    x_tensor = input.data
    d_tensor = jacobian
    for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
        orig = x_tensor[x_idx].item()
        x_tensor[x_idx] = orig - eps
        outa = fn(input).clone()
        x_tensor[x_idx] = orig + eps
        outb = fn(input).clone()
        x_tensor[x_idx] = orig
        r = (outb - outa) / (2 * eps)
        d_tensor[d_idx] = r.detach().reshape(-1)

    return jacobian


def expect_image_file_and_render_are_near(test_instance,
                                          baseline_path,
                                          result_image,
                                          max_outlier_fraction=0.001,
                                          pixel_error_threshold=0.01):
    """Compares the output of mesh_renderer with an image on disk.

    The comparison is soft: the images are considered identical if at most
    max_outlier_fraction of the pixels differ by more than a relative error of
    pixel_error_threshold of the full color value. Note that before comparison,
    mesh renderer values are clipped to the range [0,1].

    Uses _images_are_near for the actual comparison.

    Args:
      test_instance: a python unittest.TestCase instance.
      baseline_path: path to the reference image on disk.
      result_image: the result image, as a Tensor.
      max_outlier_fraction: the maximum fraction of outlier pixels allowed.
      pixel_error_threshold: pixel values are considered different if their
        difference exceeds this amount. Range is 0.0 - 1.0.
    """
    baseline_image = io.imread(baseline_path)

    test_instance.assertEqual(baseline_image.shape, result_image.shape,
                              "Images shapes {}and {} do not match."
                              .format(baseline_image.shape, result_image.shape))

    result_image = result_image.numpy()
    result_image = np.clip(result_image, 0., 1.).copy(order="C")
    baseline_image = baseline_image.astype(float) / 255.0

    diff_image = np.abs(baseline_image - result_image)
    outlier_channels = diff_image > pixel_error_threshold
    outlier_pixels = np.any(outlier_channels, axis=2)
    outlier_count = np.count_nonzero(outlier_pixels)
    outlier_fraction = outlier_count / np.prod(baseline_image.shape[:2])
    images_match = outlier_fraction <= max_outlier_fraction

    outputs_dir = "/tmp"  # os.environ["TEST_TMPDIR"]
    base_prefix = os.path.splitext(os.path.basename(baseline_path))[0]
    result_output_path = os.path.join(outputs_dir, base_prefix + "_result.png")
    diff_output_path = os.path.join(outputs_dir, base_prefix + "_diff.png")

    message = ("{} does not match. ({} of pixels are outliers, {} is allowed.)."
               " Result image written to {}, Diff written to {}"
               .format(
                   baseline_path, outlier_fraction,
                   max_outlier_fraction, result_output_path, diff_output_path))

    if not images_match:
        io.imsave(result_output_path, (result_image * 255.0).astype(np.uint8))
        diff_image[:,:,3] = 1.0
        io.imsave(diff_output_path, (diff_image * 255.0).astype(np.uint8))

    test_instance.assertTrue(images_match, msg=message)
