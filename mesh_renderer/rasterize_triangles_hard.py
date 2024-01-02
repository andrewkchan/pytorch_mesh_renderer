import torch

import rasterize_triangles_cpp


class BarycentricRasterizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, clip_space_vertices, triangles, image_width, image_height):
        """Rasterize the input mesh expressed in clip-space (xyzw) coordinates.

        Interpolates barycentric coordinates using perspective-correct interpolation
        and clips triangles that lie outside the viewing frustum.

        Args:
            clip_space_vertices: 2D float32 tensor of homogeneous vertices (xyzw)
                with shape [vertex_count, 4].
            triangles: 2D int32 tensor with shape [triangle_count, 3]. Each triplet
                should contain vertex indices describing a triangle such that the
                triangle's normal points toward the viewer if the forward order of
                the triplet defines a clockwise winding of the vertices. Gradients
                with respect to this tensor are not available.
            image_width: int specifying desired output image width in pixels.
            image_height: int specifying desired output image height in pixels.

        Returns:
            px_triangle_ids: A 2D tensor with shape [image_height, image_width].
              At return, each pixel contains a triangle id in the range
              [0, triangle_count). The id value is also 0 if there is no triangle
              at the pixel. The px_barycentric_coordinates must be checked to distinguish
              between the two cases.
            px_barycentric_coordinates: A 3D tensor with
              shape [image_height, image_width, 3]. At return, contains the triplet of
              barycentric coordinates at each pixel in the same vertex ordering as
              triangles. If no triangle is present, all coordinates are 0.
            z_buffer: A 2D tensor with shape [image_height, image_width] elements. At
              return, contains the normalized device Z coordinates of the rendered
              triangles.
        """
        px_triangle_ids, px_barycentric_coords, z_buffer = rasterize_triangles_cpp.forward(
            clip_space_vertices, triangles, image_width, image_height)
        ctx.save_for_backward(clip_space_vertices, triangles,
                              px_triangle_ids, px_barycentric_coords)
        return px_triangle_ids, px_barycentric_coords, z_buffer

    @staticmethod
    def backward(ctx, _, df_dbarycentric_coords, __):
        """Get the gradient of a scalar loss function w.r.t. input vertices
        expressed in clip-space (xyzw) coordinates.
        In the backward pass we receive a Tensor containing the gradient of the
        loss function w.r.t. our barycentric coordinate output and compute
        the gradient of the loss w.r.t. each vertex.

        Gradients w.r.t. triangle_ids or image width or height are not available.
        """
        clip_space_vertices, triangles, px_triangle_ids, px_barycentric_coords = ctx.saved_tensors
        output = rasterize_triangles_cpp.backward(
            df_dbarycentric_coords,
            clip_space_vertices,
            triangles,
            px_triangle_ids,
            px_barycentric_coords)
        df_dvertices, = output
        return df_dvertices, torch.zeros_like(triangles), None, None