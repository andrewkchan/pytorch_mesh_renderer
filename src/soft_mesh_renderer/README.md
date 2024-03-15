# soft_mesh_renderer

This package contains a differentiable, 3D mesh renderer using the probabilistic rasterization formulation by [Liu et al. 2019 "Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning"](https://arxiv.org/abs/1904.01786). It is an alternate implementation of [SoftRas](https://github.com/ShichenLiu/SoftRas) that I built for my own learning. Compare also the implementation from [PyTorch3D](https://github.com/facebookresearch/pytorch3d).

The renderer supports rendering textured triangle meshes to images with diffuse phong shading including multiple lights. Gradients of the image RGBA pixels can be obtained with respect to mesh vertices, texture colors, camera parameters, and lights.

The code is un-optimized as it's Python-only compared to the original which implements forward and backwards passes with dedicated CUDA kernels, but I hope it's more readable and others will find it useful.

# Testing

Run from the repository root:
```
python -m src.soft_mesh_renderer.test_rasterize
```

# Usage

## Rendering a shaded mesh

Rendering a shaded mesh can be done with the `render` function in `soft_mesh_renderer/render.py`. This function renders an input scene (mesh, lights, and camera) using phong shading, and returns an output image.

#### Args:

- `vertices`: 3D float32 tensor with shape `[batch_size, vertex_count, 3]`. Each triplet is an xyz position in world space.
- `triangles`: 2D int32 tensor with shape `[triangle_count, 3]`. Each triplet should contain vertex indices describing a triangle such that the triangle's normal points toward the viewer if the forward order of the triplet defines a counter-clockwise winding of the vertices. Gradients with respect to this tensor are not available.
- `diffuse_colors`: 3D float32 tensor with shape `[batch_size, vertex_count, 3]`. The RGB diffuse reflection in the range `[0, 1]` for each vertex.
- `camera_position`: 2D tensor with shape `[batch_size, 3]` or 1D tensor with shape `[3]` specifying the XYZ world space camera position.
- `camera_lookat`: 2D tensor with shape `[batch_size, 3]` or 1D tensor with shape `[3]` containing an XYZ point along the center of the camera's gaze.
- `camera_up`: 2D tensor with shape `[batch_size, 3]` or 1D tensor with shape
`[3]` containing the up direction for the camera. The camera will have no tilt with respect to this direction.
- `light_positions`: a 3D tensor with shape `[batch_size, light_count, 3]`. The world space XYZ position of each light in the scene.
- `light_intensities`: a 3D tensor with shape `[batch_size, light_count]`. The intensity values for each light. Intensities may be above 1.
- `image_width`: int specifying desired output image width in pixels.
- `image_height`: int specifying desired output image height in pixels.
- `sigma_val`: parameter controlling the sharpness of the coverage distribution for a single triangle. A smaller sigma leads to a sharper distribution.
- `gamma_val`: temperature parameter controlling uniformity of the triangle probability distribution for a pixel in the depth aggregation. When gamma is 0, all probability mass will fall into the triangle with highest z, matching the behavior of z-buffering.
- `fov_y`: float, 0D tensor, or 1D tensor with shape `[batch_size]` specifying desired output image y field of view in degrees.
- `near_clip`: float, 0D tensor, or 1D tensor with shape `[batch_size]` specifying near clipping plane distance.
- `far_clip`: float, 0D tensor, or 1D tensor with shape `[batch_size]` specifying far clipping plane distance.

#### Returns:

A 4D float32 tensor of shape `[batch_size, image_height, image_width, 4]` containing the lit RGBA color values for each image at each pixel.
- The RGB values are aggregated per-pixel according to the color aggregation formula in [1].
- The alpha values are aggregated per-pixel according to the silhouette formula in [1].

[1] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning'

### Example

An example usage of the differentiable mesh renderer to render a teapot can be seen in [`src/examples/example1b.py`](https://github.com/andrewkchan/pytorch_mesh_renderer/blob/master/src/examples/example1b.py).