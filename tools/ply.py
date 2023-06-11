import numpy as np
import torch

from typing import Union


def save_to_ply(filename: str, verts: np.ndarray, faces: np.ndarray):
    """
    Save point clouds to ply file
    :param filename: where to save the point cloud
    :param verts: (n, 3)
    :param faces: (n, 3)
    :return:
    """
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f'element vertex {n_verts}\n')
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f'element face {n_faces}\n')
        f.write('property list uint8 int32 vertex_index\n')
        f.write("end_header\n")

        lines = []
        for vert in verts:
            line = ' '.join(map(str, vert[: 3])) + '\n'
            lines.append(line)
        for face in faces:
            line = '3 ' + ' '.join(map(str, face[: 3])) + '\n'
            lines.append(line)
        f.writelines(lines)

