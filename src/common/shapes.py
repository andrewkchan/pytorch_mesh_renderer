import torch
import numpy as np

def sphere(radius, resolution=25):
    """
    Creates a triangle mesh representing a sphere with the given radius.
    The mesh will be centered on the origin.

    Returns: A tuple (vertices, triangles, normals)
    - vertices: Float tensor of shape [num_vertices, 3] giving vertices in XYZ world space.
    - triangles: Int32 tensor of shape [num_triangles, 3] giving vertex IDs of triangles.
        The vertex IDs are ordered such that they wind CCW with respect to a viewer looking
        at the outside of the sphere.
    - normals: Float tensor of shape [num_vertices, 3] giving vertex normals in XYZ world space.
        The vectors are normalized.
    """
    # We divide the sphere in K uniform longitude (phi) intervals.
    # Each longitude line starts and ends at the vertical poles of the sphere,
    # which are special vertices. Within each line, we will insert K vertices
    # between the poles by uniformly splitting latitude (theta).
    #
    # Thus, within the latitude lines not including the poles, we have equatorial
    # strips which are also uniformly split by the longitude lines. Each adjacent
    # pair of (theta, theta + theta_step) defines an equatorial strip, then when intersected
    # with an adjacent pair of phi, defines a quad on the surface of the sphere with
    # top-left corner at (theta, phi) and bottom-right corner at (theta + theta_step,
    # phi + phi_step). These quads are further split into 2 triangles each.
    #
    # The poles then connect to the adjacent latitude lines via the longitude lines.
    # Each pair of adjacent longitude lines (phi) forms a triangle.
    K = resolution
    phi_step = 2.*np.pi/K
    theta_step = np.pi/(K+1)
    num_vertices = K**2 + 2
    num_triangles = (2 * (K - 1)*K) + 2 * K
    vertices = torch.zeros([num_vertices, 3], dtype=torch.float32)
    triangles = torch.zeros([num_triangles, 3], dtype=torch.int32)
    i = 0
    # Vertex ids are grouped by latitude line:
    # 0..K-1 are theta == 1*theta_step
    # K..2K-1 are theta == 2*theta_step
    # ...
    # (K-1)*K..K*K-1 are theta == K*theta_step
    for theta in np.linspace(theta_step, np.pi - theta_step, K, endpoint=True):
        for phi in np.linspace(0., 2.*np.pi, K, endpoint=False):
            vertices[i] = radius * torch.tensor([
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
                np.sin(theta) * np.cos(phi),
            ])
            i += 1
    # Last 2 vertex ids are the poles
    vertices[num_vertices - 2] = torch.tensor([0., 1., 0.])
    vertices[num_vertices - 1] = torch.tensor([0., -1., 0.])

    triangle_id = 0
    for i in range(K-1):
        for j in range(K):
            top_left = i * K + j
            top_right = i * K + j + 1
            bottom_left = (i + 1) * K + j
            bottom_right = (i + 1) * K + j + 1
            triangles[triangle_id] = torch.tensor([top_left, bottom_left, top_right])
            triangles[triangle_id + 1] = torch.tensor([top_right, bottom_left, bottom_right])
            triangle_id += 2
    # connect top pole to topmost latitude line
    for i in range(K):
        left = i
        right = i+1
        top = num_vertices - 2
        triangles[triangle_id] = torch.tensor([top, left, right])
        triangle_id += 1
    # connect bottom pole to bottommost latitude line
    for i in range(K):
        left = (K-1)*K + i
        right = (K-1)*K + i+1
        bottom = num_vertices - 1
        triangles[triangle_id] = torch.tensor([bottom, right, left])
        triangle_id += 1
    normals = torch.nn.functional.normalize(vertices, p=2.0, dim=-1)
    return vertices, triangles, normals

def cube(size):
    """
    Creates a triangle mesh representing a cube with the given side length.
    The mesh will be centered on the origin.

    Returns: A tuple (vertices, triangles, normals)
    - vertices: Float tensor of shape [num_vertices, 3] giving vertices in XYZ world space.
    - triangles: Int32 tensor of shape [num_triangles, 3] giving vertex IDs of triangles.
        The vertex IDs are ordered such that they wind CCW with respect to a viewer looking
        at the outside of the sphere.
    - normals: Float tensor of shape [num_vertices, 3] giving vertex normals in XYZ world space.
        The vectors are normalized. Note that face-vertex normals are not supported and so
        the vertex normals will be the average of the normals of the incident faces.
    """
    vertices = 0.5 * size * torch.tensor(
        [[-1, -1, 1], [-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [1, -1, 1],
        [1, -1, -1], [1, 1, -1], [1, 1, 1]],
        dtype=torch.float32)
    normals = torch.nn.functional.normalize(vertices, p=2.0, dim=-1)
    triangles = torch.tensor(
        [
            [2, 1, 0],
            [0, 3, 2],
            [6, 2, 3],
            [3, 7, 6],
            [5, 6, 7],
            [7, 4, 5],
            [1, 5, 4],
            [4, 0, 1],
            [2, 6, 5],
            [5, 1, 2],
            [0, 4, 7],
            [7, 3, 0]
        ],
        dtype=torch.int32)
    return vertices, triangles, normals