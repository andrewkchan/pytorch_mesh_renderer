from mesh_renderer import camera_utils
import torch
import math

"""
rasterize_triangles_soft.py

Implements BarycentricRasterizer with PyTorch-only primitives (no C++ extensions).
"""

# Returns a 4x4 viewport matrix which can be used to convert 3D homogeneous points in clip space to screen space,
# specified by args:
# - x: integer giving the screen space X offset
# - y: integer giving the screen space Y offset
# - w: integer giving the total screen space width
# - h: integer giving the total screen space height
# - z_buffer_res: number giving resolution of the z-buffer
#
# The bi-unit cube [-1, 1]*[-1, 1]*[-1, 1] should be mapped onto the screen cube [x, x+w]*[y, y+h]*[0, z_buffer_res].
def viewport(x, y, w, h, z_buffer_res):
    res = torch.eye(4)
    res[0][0] = w/2.0
    res[1][1] = h/2.0
    res[2][2] = z_buffer_res/2.0
    res[0:3, 3] = torch.tensor([x + w/2.0, y + h/2.0, z_buffer_res / 2.0])
    return res

# Returns barycentric coordinates of a 3D point P w.r.t. triangle v0, v1, v2.
# The input `M_inv` should be the inverse of a 3x3 matrix where the columns are the vertices.
def barycentric(M_inv, p):
    return M_inv @ p

def rasterize_barycentric(clip_space_vertices, triangles, image_width, image_height):
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
    z_buffer = torch.ones([image_height, image_width], dtype=torch.float32)
    px_triangle_ids = torch.zeros([image_height, image_width], dtype=torch.int32)
    px_barycentric_coordinates = torch.zeros([image_height, image_width, 3], dtype=torch.float32)

    # z-buffer ranges from 0.0 to 1.0, anything outside gets clipped
    z_buffer_res = 1.0
    viewport_mat = viewport(0., 0., image_width, image_height, z_buffer_res)
    px_M = torch.zeros(3, 3)

    for triangle_id in range(len(triangles)):
        triangle = triangles[triangle_id]
        proj_v012 = clip_space_vertices[triangle] # shape: [3, 4]
        proj_v012_w = proj_v012[:,[3]] # shape: [3, 1]

        # clip space to screen space
        px_v012 = (viewport_mat @ (proj_v012 / (proj_v012_w)).T).T[:,:3]

        # get bbox in screen-space
        minx = math.floor(
            max(0, min(px_v012[0][0], px_v012[1][0], px_v012[2][0], image_width))
        )
        miny = math.floor(
            max(0, min(px_v012[0][1], px_v012[1][1], px_v012[2][1], image_height))
        )
        maxx = math.ceil(
            min(image_width, max(px_v012[0][0], px_v012[1][0], px_v012[2][0], 0))
        )
        maxy = math.ceil(
            min(image_height, max(px_v012[0][1], px_v012[1][1], px_v012[2][1], 0))
        )

        px_M[:] = px_v012.T
        px_M[2,:] = torch.tensor([1., 1., 1.])
        try:
            px_M_inv = px_M.inverse()
        except Exception:
            # Screen-space vertex basis is not invertible, meaning triangle is
            # degenerate when projected (zero area). Skip rendering
            continue

        did_draw = 0
        # Depths of the screen-space vertices as suitable for z-test.
        # Note that depth is inversely proportional to the vertex eye-space z-coordinate.
        vertex_depths = px_v012[:,2]

        for y in range(miny, maxy):
            if y<0 or y>=image_height:
                continue
            for x in range(minx, maxx):
                if x<0 or x>=image_width:
                    continue
                p = torch.tensor([x + 0.5, y + 0.5, 1.])
                bc_screen = barycentric(px_M_inv, p)
                if bc_screen[0] < 0 or bc_screen[1] < 0 or bc_screen[2] < 0:
                    # pixel is not inside triangle
                    continue
                else:
                    # get perspective-correct barycentric coordinates
                    bc = torch.nn.functional.normalize(bc_screen / proj_v012_w.T[0], dim=0, p=1)
                    z = vertex_depths @ bc_screen
                    if z < 0.0 or z > 1.0 or z > z_buffer[y][x]:
                        continue
                    did_draw += 1
                    z_buffer[y][x] = z
                    px_triangle_ids[y][x] = triangle_id
                    px_barycentric_coordinates[y][x] = bc
        print("drew {} pixels for triangle {}".format(did_draw, triangle_id))

    return px_triangle_ids, px_barycentric_coordinates, z_buffer
