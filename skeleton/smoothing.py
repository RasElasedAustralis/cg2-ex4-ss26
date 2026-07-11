import numpy as np

def build_neighbors(n_vertices, faces):
    neighbors = [set() for _ in range(n_vertices)]

    for a, b, c in faces:
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])

    return neighbors

def laplacian_smooth(vertices, neighbors, h=1.0):
    new_vertices = vertices.copy()

    for i, nbrs in enumerate(neighbors):
        if len(nbrs) == 0:
            continue

        avg = np.mean(vertices[list(nbrs)], axis=0)
        new_vertices[i] = (1 - h) * vertices[i] + h * avg

    return new_vertices

def smooth_n_times(n, vertices, faces, h=1.0):
    neighbors = build_neighbors(len(vertices), faces)

    for _ in range(n):
        vertices = laplacian_smooth(vertices, neighbors, h)

    return vertices