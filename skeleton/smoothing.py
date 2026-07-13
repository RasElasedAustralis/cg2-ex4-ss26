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

def cotangent(vi, vj, vk):
    u = vj - vi
    v = vk - vi
    
    cross = np.linalg.norm(np.cross(u, v))
    
    if cross < 1e-12:
        return 0.0
    
    return np.dot(u, v) / cross
    
# def cotangent(vi, vj, vk):
#     u = vj - vi
#     v = vk - vi
    
#     nu = np.linalg.norm(u)
#     nv = np.linalg.norm(v)
#     if nu < 1e-12 or nv < 1e-12:
#         return 0.0

#     cos_angle = np.dot(u, v) / (nu * nv)
#     sin_angle = np.linalg.norm(np.cross(u, v) / (nu * nv)) 
#     cotan = cos_angle / sin_angle if sin_angle > 1e-12 else 0.0
#     # if cotan < 0:
#     #     print(f"Warning: negative cotangent value {cotan} for triangle with vertices {vi}, {vj}, {vk}")
#     #     cotan = 0.0
#     return cotan

def build_cotan_laplacian(vertices, faces):
    L = np.zeros((len(vertices), len(vertices)))
    # max_weight = 0
    # min_weight = float('inf')
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        cot_i = cotangent(vi, vj, vk)
        cot_j = cotangent(vj, vk, vi)
        cot_k = cotangent(vk, vi, vj)
        w_ij = cot_k / 2
        w_jk = cot_i / 2
        w_ik = cot_j / 2
        # max_weight = max(max_weight, cot_i, cot_j, cot_k)
        # min_weight = min(min_weight, cot_i, cot_j, cot_k)
        
        L[i, j] -= w_ij
        L[j, i] -= w_ij
        L[j, k] -= w_jk
        L[k, j] -= w_jk
        L[i, k] -= w_ik
        L[k, i] -= w_ik
        
        L[i, i] += w_ij + w_ik
        L[j, j] += w_ij + w_jk
        L[k, k] += w_jk + w_ik
    # print("max cotan weight:", max_weight)
    # print("min cotan weight:", min_weight)
    return L

def build_mass_matrix(vertices, faces):
    mass = np.zeros(len(vertices))
    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        mass[i] += 1/3 * area
        mass[j] += 1/3 * area
        mass[k] += 1/3 * area

    M_inv = np.zeros((len(vertices), len(vertices)))
    for i in range(len(vertices)):
        if mass[i] > 0:
            M_inv[i, i] = 1.0 / mass[i]
            
    return M_inv

def laplacian_smooth_cotan(vertices, L, M_inv, h=0.1):
    new_vertices = vertices.copy()
    Lx = np.matmul(L, vertices)
    delta = np.matmul(M_inv, Lx)
    new_vertices = vertices - h * delta
    return new_vertices

def smooth_n_times_cotan(n, vertices, faces, h=0.1):
    # print("n:", n)
    # print("h:", h)
    for _ in range(n):
        L = build_cotan_laplacian(vertices, faces)
        M_inv = build_mass_matrix(vertices, faces)
        # print("max M_inv:", np.max(M_inv))
        # print("min mass:", np.min(1 / np.diag(M_inv)))
        # print("max mass:", np.max(1 / np.diag(M_inv))) 
        vertices = laplacian_smooth_cotan(vertices, L, M_inv, h)

    return vertices