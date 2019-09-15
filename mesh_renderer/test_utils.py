from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from skimage import io
import torch


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

    outlier_channels = (np.abs(baseline_image - result_image) > pixel_error_threshold)
    outlier_pixels = np.any(outlier_channels, axis=2)
    outlier_count = np.count_nonzero(outlier_pixels)
    outlier_fraction = outlier_count / np.prod(baseline_image.shape[:2])
    images_match = outlier_fraction <= max_outlier_fraction

    outputs_dir = "/tmp" # os.environ["TEST_TMPDIR"]
    base_prefix = os.path.splitext(os.path.basename(baseline_path))[0]
    result_output_path = os.path.join(outputs_dir, base_prefix + "_result.png")

    message = ("{} does not match. ({} of pixels are outliers, {} is allowed.). "
               "Result image written to {}"
               .format(baseline_path, outlier_fraction, max_outlier_fraction, result_output_path))

    if not images_match:
        io.imsave(result_output_path, result_image * 255.0)

    test_instance.assertTrue(images_match, msg=message)
