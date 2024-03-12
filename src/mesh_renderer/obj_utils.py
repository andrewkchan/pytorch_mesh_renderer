import os

import torch

def load_obj(filename, normalize=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x), normals (vn x x x), 
    and faces (f x x x). Per-face-vertex normals are not supported;
    they will be averaged out so that each vertex gets exactly 1 normal.

    Returns:
    - vertices, faces, normals: Tuple of tensors with shapes 
        ([vertex_count, 3], [triangle_count, 3], [vertex_count, 3])
        and types (float32, int32, float32).
    """

    vertices = []
    all_normals = []
    vertex_id_to_normals = {}
    faces = []
    with open(filename) as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue
        if parts[0] == 'v':
            vertices.append([float(v) for v in parts[1:4]])
        elif parts[0] == 'vn':
            all_normals.append([float(v) for v in parts[1:4]])
        elif parts[0] == 'f':
            face_vertices = line.split()[1:]
            if len(face_vertices) > 3:
                print("warning: encountered a face with more than 3 vertices," +
                    "extra vertices will be skipped")
            faces.append([int(face_vertex.split('/')[0]) for face_vertex in face_vertices[:3]])
            for face_vertex in face_vertices[:3]:
                parts = face_vertex.split('/')
                vertex_id = int(parts[0]) - 1
                normal_id = int(parts[2]) - 1
                if vertex_id not in vertex_id_to_normals:
                    vertex_id_to_normals[vertex_id] = []
                vertex_id_to_normals[vertex_id].append(normal_id)
    
    vertices = torch.tensor(vertices, dtype=torch.float32)
    all_normals = torch.tensor(all_normals, dtype=torch.float32)
    normals = torch.zeros_like(vertices)
    # average all face-vertex normals to a single normal vector per vertex
    for i in range(len(vertices)):
        if i not in vertex_id_to_normals:
            normals[i] = torch.ones(3)
            continue
        n = len(vertex_id_to_normals[i])
        for j in vertex_id_to_normals[i]:
            normals[i] += all_normals[j] / n
    # normalize normal vectors
    normals = torch.nn.functional.normalize(normals, p=2.0, dim=1)
    faces = torch.tensor(faces, dtype=torch.int32) - 1

    if normalize:
        # normalize into a unit cube centered around zero
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2
    
    return vertices, faces, normals
    