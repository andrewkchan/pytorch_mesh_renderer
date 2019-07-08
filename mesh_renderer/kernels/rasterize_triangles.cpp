#include <algorithm>
#include <vector>

#include <torch/extension.h>

typedef int int32;
typedef long long int64;

// Compute the triangle id, barycentric coordinates, and z-buffer at each pixel
// in the image.
//
// vertices: A flattened 2D array with 4*vertex_count elements.
//   Each contiguous triplet is the XYZW location of the vertex with that
//   triplet's id. The coordinates are assumed to be OpenGL-style clip-space
//   (i.e., post-projection, pre-divide), where X points right, Y points up,
//   Z points away.
// triangles: A flattened 2D array with 3*triangle_count elements.
//   Each contiguous triplet is the three vertex ids indexing into vertices
//   describing one triangle with clockwise winding.
// triangle_count: The number of triangles stored in the array triangles.
// px_triangle_ids: A flattened 2D array with image_height*image_width elements.
//   At return, each pixel contains a triangle id in the range
//   [0, triangle_count). The id value is also 0 if there is no triangle
//   at the pixel. The px_barycentric_coordinates must be checked to distinguish
//   between the two cases.
// px_barycentric_coordinates: A flattened 3D array with
//   image_height*image_width*3 elements. At return, contains the triplet of
//   barycentric coordinates at each pixel in the same vertex ordering as
//   triangles. If no triangle is present, all coordinates are 0.
// z_buffer: A flattened 2D array with image_height*image_width elements. At
//   return, contains the normalized device Z coordinates of the rendered
//   triangles.
std::vector<torch::Tensor> rasterize_triangles_forward(
  const float* vertices,
  const int32* triangles,
  int32 triangle_count,
  int32 image_width,
  int32 image_height,
  int32* px_triangle_ids,
  float* px_barycentric_coordinates,
  float* z_buffer
) {
  // pass
}
