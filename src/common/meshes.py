import torch

def compute_vertex_normals(vertices, triangles):
    """
    Computes vertex normals for a triangle mesh by first computing
    face normals, then averaging the normals on incident vertices.
    The resulting vectors are normalized.

    Args:
      vertices: 3D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
      triangles: 2D int32 tensor with shape [triangle_count, 3].

    Returns:
    - A tensor with shape [batch_size, vertex_count, 3] providing per-vertex normal
      vectors.
    """
    batch_size = vertices.shape[0]
    normals = torch.zeros_like(vertices)
    for b in range(batch_size):
        # vertices_faces[i][j] gives the vertex corresponding to triangles[i][j]
        vertices_faces = vertices[b, triangles.long(), :] # [vertex_count, 3, 3]
        normals[b].index_add_(0, triangles[:, 0].long(),
            torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                        vertices_faces[:, 2] - vertices_faces[:, 0])
        )
        normals[b].index_add_(0, triangles[:, 1].long(),
            torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                        vertices_faces[:, 0] - vertices_faces[:, 1])
        )
        normals[b].index_add_(0, triangles[:, 2].long(),
            torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                        vertices_faces[:, 1] - vertices_faces[:, 2])
        )
    normals = torch.nn.functional.normalize(normals, eps=1e-6, p=2, dim=-1)
    return normals