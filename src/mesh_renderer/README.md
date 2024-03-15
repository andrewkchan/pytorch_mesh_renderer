# mesh_renderer

This package contains a differentiable, 3D mesh renderer using the barycentric formulation from Genova, Kyle, et al. "Unsupervised training for 3d morphable model regression." It is a port of Google's [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) to PyTorch.

There is an optimized C++ implementation of this renderer available for use. To enable it, first install the kernel via `cd src/mesh_renderer/kernel && python setup.py install`, then change the hardcoded config variable `USE_CPP_RASTERIZER` as described in the [mesh_renderer docs](https://github.com/andrewkchan/pytorch_mesh_renderer/blob/master/src/mesh_renderer/README.md).

# Testing

To test the rasterizer module, run from the repository root:
```
python -m src.mesh_renderer.rasterize_triangles_test
```

To test the mesh renderer, run from the repository root:

```
python -m src.mesh_renderer.mesh_renderer_test
```

# Usage

The mesh renderer provides a high-level API for rendering triangle meshes with shading and a low-level API for rasterizing batches of triangles. The APIs are mostly the same as those in [tf_mesh_renderer](https://github.com/google/tf_mesh_renderer) but adjusted for PyTorch.

## Rendering a shaded mesh

### `mesh_renderer`

Rendering a shaded mesh can be done with the `mesh_renderer` function in `mesh_renderer/mesh_renderer.py`. This function renders an input scene (mesh, lights, and camera) using phong shading, and returns an output image.

#### Args:

- `vertices`: 3D float32 tensor with shape `[batch_size, vertex_count, 3]`. Each triplet is an xyz position in world space.
- `triangles`: 2D int32 tensor with shape `[triangle_count, 3]`. Each triplet should contain vertex indices describing a triangle such that the triangle's normal points toward the viewer if the forward order of the triplet defines a clockwise winding of the vertices. Gradients with respect to this tensor are not available.
- `normals`: 3D float32 tensor with shape `[batch_size, vertex_count, 3]`. Each triplet is the xyz vertex normal for its corresponding vertex. Each vector is assumed to be already normalized.
- `diffuse_colors`: 3D float32 tensor with shape `[batch_size, vertex_count, 3]`. The RGB diffuse reflection in the range `[0, 1]` for each vertex.
- `camera_position`: 2D tensor with shape `[batch_size, 3]` or 1D tensor with shape ` [3]` specifying the XYZ world space camera position.
- `camera_lookat`: 2D tensor with shape [batch_size, 3] or 1D tensor with shape `[3]` containing an XYZ point along the center of the camera's gaze.
- `camera_up`: 2D tensor with shape `[batch_size, 3]` or 1D tensor with shape `[3]` containing the up direction for the camera. The camera will have no tilt with respect to this direction.
- `light_positions`: a 3D tensor with shape `[batch_size, light_count, 3]`. The XYZ position of each light in the scene. In the same coordinate space as pixel_positions.
- `light_intensities`: a 3D tensor with shape `[batch_size, light_count, 3]`. The RGB intensity values for each light. Intensities may be above 1.
- `image_width`: int specifying desired output image width in pixels.
- `image_height`: int specifying desired output image height in pixels.
- `specular_colors`: (optional) 3D float32 tensor with shape `[batch_size, vertex_count, 3]`. The RGB specular reflection in the range `[0, 1]` for each vertex. If supplied, specular reflections will be computed, and both specular colors and shininess_coefficients are expected.
- `shininess_coefficients`: (optional) a 0D-2D float32 tensor with maximum shape `[batch_size, vertex_count]`. The phong shininess coefficient of each vertex. A 0D tensor or float gives a constant shininess coefficient of all vertices across all batches and images. A 1D tensor must have shape `[batch_size]`, and a single shininess coefficient per image is used.
- `ambient_color`: (optional) a 2D tensor with shape `[batch_size, 3]`. The RGB ambient color, which is added to each pixel in the scene. If None, it is assumed to be black.
- `fov_y`: (optional) float, 0D tensor, or 1D tensor with shape `[batch_size]` specifying desired output image y field of view in degrees.
- `near_clip`: (optional) float, 0D tensor, or 1D tensor with shape `[batch_size]` specifying near clipping plane distance.
- `far_clip`: (optional) float, 0D tensor, or 1D tensor with shape [batch_size] specifying far clipping plane distance.


#### Returns:

A 4D float32 tensor of shape `[batch_size, image_height, image_width, 4]` containing the lit RGBA color values for each image at each pixel. RGB colors are the intensity values before tonemapping and can be in the range `[0, infinity]`. Clipping to the range `[0, 1]` with `np.clip` is likely reasonable for both viewing and training most scenes. More complex scenes with multiple lights should tone map color values for display only. One simple tonemapping approach is to rescale color values as x/(1+x); gamma compression is another common technique. Alpha values are zero for background pixels and near one for mesh pixels.

### Example

An example usage of the differentiable mesh renderer to render a cube, then optimize its rotation to match a target image can be seen in the `testThatCubeRotates` test case in `mesh_renderer_test.py`.

## Rasterizing triangles with arbitrary attributes

### `rasterize`

This is a lower-level function which can be used to rasterize a batch of triangles into a tensor providing interpolated vertex attributes in each pixel. This could be useful if you want to build your own shading on top of the core rasterization module, for example.

#### Args:

- `world_space_vertices`: 3D float32 tensor of xyz positions with shape `[batch_size, vertex_count, 3]`.
- `attributes`: 3D float32 tensor with shape `[batch_size, vertex_count, attribute_count]`. Each vertex attribute is interpolated across the triangle using barycentric interpolation.
- `triangles`: 2D int32 tensor with shape `[triangle_count, 3]`. Each triplet should contain vertex indices describing a triangle such that the triangle's normal points toward the viewer if the forward order of the triplet defines a clockwise winding of the vertices. Gradients with respect to this tensor are not available.
- `camera_matrices`: 3D float tensor with shape `[batch_size, 4, 4]` containing model-view-perspective projection matrices.
- `image_width`: int specifying desired output image width in pixels.
- `image_height`: int specifying desired output image height in pixels.
- `background_value`: a 1D float32 tensor with shape `[attribute_count]`. Pixels that lie outside all triangles take this value.

#### Returns:

- A 4D float32 tensor with shape `[batch_size, image_height, image_width, attribute_count]`, containing the interpolated vertex attributes at each pixel.

### Example

An example usage of the `rasterize` API to rasterize a cube can be found in the `testRendersTwoCubesInBatch` test case in `rasterize_triangles_test.py`.

## `camera_utils`

This file contains some utilities that may be useful for transforming the input scene before rendering. The `mesh_renderer` function uses some of these functions internally to project the world-space vertices into camera-space. Model-view-perspective projection matrices are also required as input to the lower-level rasterization APIs.

### `euler_matrices`.

You can use this to create a Model matrix with rotation to transform a set of object-space vertices into world space before rendering it.

#### Args:

- `angles`: a `[batch_size, 3]` tensor containing X, Y, and Z angles in radians.

#### Returns:

- A `[batch_size, 4, 4]` tensor of matrices.

### `look_at`

You can use this to compute a View matrix to transform a set of world-space vertices into eye space; this is primarily useful for the lower-level rasterization APIs which require an input View matrix.

#### Args:

- `eye`: 2D float32 tensor with shape `[batch_size, 3]` containing the XYZ world space position of the camera.
- `center`: 2D float32 tensor with shape `[batch_size, 3]` containing a position along the center of the camera's gaze line.
- `world_up`: 2D float32 tensor with shape `[batch_size, 3]` specifying the world's up direction; the output camera will have no tilt with respect to this direction.

#### Returns:

- A `[batch_size, 4, 4]` float tensor containing a right-handed camera extrinsics matrix that maps points from world space to points in eye space.

# Implementation notes


## Rasterizer

There are two implementations of the low-level `rasterize` API.

### C++ kernel

This implementation is written in C++ for performance. Since it doesn't use PyTorch built-in functions under-the-hood and instead [extends `torch.autograd.Function`](https://pytorch.org/docs/stable/notes/extending.html#extending-autograd), the backward pass is explicitly written rather than just being implicit in the forward pass. Both are written in the [C++ extension](https://pytorch.org/tutorials/advanced/cpp_extension.html) in `src/mesh_renderer/kernels/rasterize_triangles.cpp`, with the wrapper code in `src/mesh_renderer/rasterize_triangles_ext.py`.

This implementation is enabled by setting the hard-coded global variable `USE_CPP_RASTERIZER = True` in `src/mesh_renderer/rasterize_triangles.py`.

### Python-only kernel

This implementation is written in Python only in `src/mesh_renderer/rasterize_triangles_python.py` and leverages PyTorch built-in functions for autograd. It's much shorter than the C++ kernel and is intended to be simpler to understand. However, performance is much worse.

This implementation is enabled by setting the hard-coded global variable `USE_CPP_RASTERIZER = False` in `src/mesh_renderer/rasterize_triangles.py`. This is the default.