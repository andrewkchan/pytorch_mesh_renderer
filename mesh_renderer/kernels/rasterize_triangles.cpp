#include <algorithm>
#include <cmath>
#include <vector>

#include <torch/extension.h>

namespace {

  // Threshold for a barycentric coordinate triplet's sum, below which the
  // coordinates at a pixel are deemed degenerate. Most such degenerate
  // triplets in an image will be exactly zero, as this is how pixels outside
  // the mesh are rendered.
  constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

  // If the area of a triangle is very small in screen space, the corner
  // vertices are approaching colinearity, and we should drop the gradient
  // to avoid numerical instability (in particular, blowup, as the forward
  // pass computation already only has 8 bits of precision).
  constexpr float kMinimumTriangleArea = 1e-13;
   
}

// Takes the maximum of a, b, and c, rounds up, and converts to an integer
// in the range [low, high].
inline int clamped_integer_max(float a, float b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::ceil(std::max(std::max(a, b), c))), low),
      high);
}

// Takes the minimum of a, b, and c, rounds down, and converts to an integer
// in the range [low, high].
inline int clamped_integer_min(float, a, float, b, float c, int low, int high) {
  return std::min(
      std::max(static_cast<int>(std::floor(std::min(std::min(a, b), c))), low),
      high);
}

// Compute the edge functions from M^-1 as described by Olano and Greer,
// "Triangle Scan Conversion using 2D Homogeneous Coordinates."
//
// This function combines equations (3) and (4). It first computes
// [a b c] = u_i * M^-1, where u_0 = [1 0 0], u_1 = [0 1 0], etc.,
// then computes edge_i = aX + bY + c.
void compute_edge_functions(const float px, const float py,
                            const float m_inv[9], float values[3]) {
  for (int i = 0; i < 3; ++i) {
    const float a = m_inv[3 * i + 0];
    const float b = m_inv[3 * i + 1];
    const float c = m_inv[3 * i + 2];

    values[i] = a * px + b * py + c;
  }
}

// Compute a 3x3 matrix inverse without dividing by the determinant.
// Instead, makes an unnormalized matrix inverse with the corect sign
// by flipping a sign of the matric if the determinant is negative.
// By leaving out determinant division, the rows of M^-1 only depend on two out
// of three of the columns of M; i.e., the first row of M^-1 only depends on the
// second and third columns of M, the second only depends on the first and
// third, etc. This means we can compute edge functions for two neighboring
// triangles independently and produce exactly the same numerical result up
// to the sign. This in turn means we can avoid cracks in rasterization without
// using fixed-point arithmetic.
// See http://mathworld.wolfram.com/MatrixInverse.html
void compute_unnormalized_matrix_inverse(
  const float a11, const float a12, const float a13,
  const float a21, const float a22, const float a23,
  const float a31, const float a32, const float a33, float m_inv[9]) {
  m_inv[0] = a22 * a33 - a32 * a23;
  m_inv[1] = a13 * a32 - a33 * a12;
  m_inv[2] = a12 * a23 - a22 * a13;
  m_inv[3] = a23 * a31 - a33 * a21;
  m_inv[4] = a11 * a33 - a31 * a13;
  m_inv[5] = a13 * a21 - a23 * a11;
  m_inv[6] = a21 * a32 - a31 * a22;
  m_inv[7] = a12 * a31 - a32 * a11;
  m_inv[8] = a11 * a22 - a21 * a12;

  // The first column of the unnormalized M^-1 contains intermediate values for
  // det(M).
  const float det = a11 * m_inv[0] + a12 * m_inv[3] + a13 * m_inv[6];

  // Transfer the sign of the determinant.
  if (det < 0.0f) {
    for (int i = 0; i < 9; ++i) {
      m_inv[i] = -m_inv[i];
    }
  }
}

// Determine whether the point p lies inside a front-facing triangle.
// Count pixels exactly on an edge as inside the triangle, as long as the
// triangle is not degenerate. Degenerate (zero-area) triangles always fail
// the inside test.
bool pixel_is_inside_triangle(const float edge_values[3]) {
  // Check that the edge values are all non-negative and that at least one is
  // positive (triangle is non-degenerate).
  return (edge_values[0] >= 0 && edge_values[1] >= 0 && edge_values[2] >= 0) &&
         (edge_values[0] > 0 || edge_values[1] > 0 || edge_values[2] > 0);
}

std::vector<torch::Tensor> rasterize_triangles_backward() {
  return {}; // TODO.
}

// Compute the triangle id, barycentric coordinates, and z-buffer at each pixel
// in the image.
//
// Params:
// vertices: A flattened 2D array with 4*vertex_count elements.
//   Each contiguous triplet is the XYZW location of the vertex with that
//   triplet's id. The coordinates are assumed to be OpenGL-style clip-space
//   (i.e., post-projection, pre-divide), where X points right, Y points up,
//   Z points away.
// triangles: A flattened 2D array with 3*triangle_count elements.
//   Each contiguous triplet is the three vertex ids indexing into vertices
//   describing one triangle with clockwise winding.
// triangle_count: The number of triangles stored in the array triangles.
//
// Returns:
// px_triangle_ids: A 2D tensor with shape {image_height, image_width}.
//   At return, each pixel contains a triangle id in the range
//   [0, triangle_count). The id value is also 0 if there is no triangle
//   at the pixel. The px_barycentric_coordinates must be checked to distinguish
//   between the two cases.
// px_barycentric_coordinates: A 3D tensor with
//   shape {image_height, image_width, 3}. At return, contains the triplet of
//   barycentric coordinates at each pixel in the same vertex ordering as
//   triangles. If no triangle is present, all coordinates are 0.
// z_buffer: A 2D tensor with shape {image_height, image_width} elements. At
//   return, contains the normalized device Z coordinates of the rendered
//   triangles.
std::vector<torch::Tensor> rasterize_triangles_forward(
  const torch::Tensor &vertices,
  const torch::Tensor &triangles,
  int triangle_count,
  int image_width,
  int image_height
) {
  const float half_image_width = 0.5 * image_width;
  const float half_image_height = 0.5 * image_height;
  float unnormalized_matrix_inverse[9];
  float b_over_w[3];
  auto px_triangle_ids = torch::zeros(
    {image_height, image_width},
    torch::TensorOptions().dtype(torch::kInt32));
  auto px_barycentric_coords = torch::zeros(
    {image_height, image_width, 3},
    torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  auto z_buffer = torch::ones(
    {image_height, image_width},
    torch::TensorOptions().dtype(torch::kFloat32));

  auto vertices_a = vertices.accessor<float, 2>();
  auto triangles_a = triangles.accessor<int, 2>();
  auto z_buffer_a = z_buffer.accessor<float, 2>();
  auto px_triangle_ids_a = px_triangle_ids.accessor<int, 2>();
  auto px_barycentric_coordinates_a =
    px_barycentric_coordinates.accessor<float, 3>();

  for (int triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
    const int v0_id = 4 * triangles_a[triangle_id][0];
    const int v1_id = 4 * triangles_a[triangle_id][1];
    const int v2_id = 4 * triangles_a[triangle_id][2];

    const float v0w = vertices_a[v0_id][3];
    const float v1w = vertices_a[v1_id][3];
    const float v2w = vertices_a[v2_id][3];
    // Early exit: if all w < 0, triangle is entirely behind the eye.
    if (v0w < 0 && v1w < 0 && v2w < 0) {
      continue;
    }

    const float v0x = vertices_a[v0_id][0];
    const float v0y = vertices_a[v0_id][1];
    const float v1x = vertices_a[v1_id][0];
    const float v1y = vertices_a[v1_id][1];
    const float v2x = vertices_a[v2_id][0];
    const float v2y = vertices_a[v2_id][1];

    compute_unnormalized_matrix_inverse(v0x, v1x, v2x,
                                        v0y, v1y, v2y,
                                        v0w, v1w, v2w,
                                        unnormalized_matrix_inverse);

    // Initialize the bounding box to the entire screen.
    int left = 0, right = image_width, bottom = 0, top = image_height;
    // If the triangle is entirely inside the screen, project the vertices to
    // pixel coordinates and find the triangle bounding box enlarged to the
    // nearest integer and clamped to the image boundaries.
    if (v0w > 0 && v1w > 0 && v2w > 0) {
      const float p0x = (v0x / v0w + 1.0) * half_image_width;
      const float p1x = (v1x / v1w + 1.0) * half_image_width;
      const float p2x = (v2x / v2w + 1.0) * half_image_width;
      const float p0y = (v0y / v0w + 1.0) * half_image_height;
      const float p1y = (v1y / v1w + 1.0) * half_image_height;
      const float p2y = (v2y / v2w + 1.0) * half_image_height;
      left = clamped_integer_min(p0x, p1x, p2x, 0, image_width);
      right = clamped_integer_max(p0x, p1x, p2x, 0, image_width);
      bottom = clamped_integer_min(p0y, p1y, p2y, 0, image_height);
      top = clamped_integer_max(p0y, p1y, p2y, 0, image_height);
    }

    // Iterate over each pixel in the bounding box.
    for (int iy = bottom; iy < top; ++iy) {
      for (int ix = left; ix < right; ++ix) {
        const float px = ((ix + 0.5) / half_image_width) - 1.0;
        const float py = ((iy + 0.5) / half_image_height) - 1.0;

        compute_edge_functions(px, py, unnormalized_matrix_inverse, b_over_w);
        if (!pixel_is_inside_triangle(b_over_w)) {
          continue;
        }

        const float one_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2];
        const float b0 = b_over_w[0] / one_over_w;
        const float b1 = b_over_w[1] / one_over_w;
        const float b2 = b_over_w[2] / one_over_w;

        const float v0z = vertices_a[v0_id][2];
        const float v1z = vertices_a[v1_id][2];
        const float v2z = vertices_a[v2_id][2];
        // Since we computed an unnormalized w above, we need to recompute
        // a properly scaled clip-space w value and then divide clip-space z
        // by that.
        const float clip_z = b0 * v0z + b1 * v1z + b2 * v2z;
        const float clip_w = b0 * v0w + b1 * v1w + b2 * v2w;
        const float z = clip_z / clip_w;

        // Skip the pixel if it is farther than the current z-buffer pixel or
        // beyond the near or far clipping plane.
        if (z < -1.0 || z > 1.0 || z > z_buffer_a[iy][ix]) {
          continue;
        }

        px_triangle_ids_a[iy][ix] = triangle_id;
        z_buffer_a[iy][ix] = z;
        px_barycentric_coordinates_a[iy][ix][0] = b0;
        px_barycentric_coordinates_a[iy][ix][1] = b1;
        px_barycentric_coordinates_a[iy][ix][2] = b2;
      }
    }
  }

  return {
    px_triangle_ids,
    px_barycentric_coordinates,
    z_buffer
  };
}
