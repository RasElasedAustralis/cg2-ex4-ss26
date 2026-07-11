from dataclasses import dataclass
from pathlib import Path
import pywavefront as pw

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PointCloud:
    vertices: np.ndarray
    normals: np.ndarray
    kdtree: cKDTree
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_diagonal: float
    name: str

@dataclass
class ObjectMesh:
    vertices: np.ndarray
    faces: np.ndarray
    kdtree: cKDTree
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_diagonal: float
    name: str

def load_obj_file(path: Path):
    scene = pw.Wavefront(str(path), collect_faces=True, parse=True)
    vertices = np.array(scene.vertices, dtype=float)
    #print(vars(scene))
    print(scene.mesh_list[0])

    return ObjectMesh(
        vertices=vertices,
        faces=np.array(scene.mesh_list[0].faces, dtype=int),
        kdtree=cKDTree(vertices),
        bbox_min=vertices.min(axis=0),
        bbox_max=vertices.max(axis=0),
        bbox_diagonal=np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)),
        name=path.name
    )

def load_point_cloud(path: Path) -> PointCloud:
    with open(path, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    n_vertices, _, _ = map(int, lines[1].split()[:3])

    vertex_lines = lines[2 : 2 + n_vertices]

    vertices = []
    normals = []

    for line in vertex_lines:
        values = list(map(float, line.split()))
        vertices.append(values[:3])
        normals.append(values[3:6])

    vertices = np.array(vertices, dtype = float)
    normals = np.array(normals, dtype = float)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)

    return PointCloud(
        vertices = vertices,
        normals = normals,
        kdtree = cKDTree(vertices),
        bbox_min = bbox_min,
        bbox_max = bbox_max,
        bbox_diagonal = bbox_diagonal,
        name = path.name
    )