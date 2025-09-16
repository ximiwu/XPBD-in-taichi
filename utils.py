import taichi as ti
import taichi.math as tm
import numpy as np

vec2f = ti.types.vector(2, ti.f32)
vec3f = ti.types.vector(3, ti.f32)
vec4f = ti.types.vector(4, ti.f32)

vec2i = ti.types.vector(2, ti.i32)
vec3i = ti.types.vector(3, ti.i32)
vec4i = ti.types.vector(4, ti.i32)


def gen_cube(scale=(1.0, 1.0, 1.0), rotation_angles=(0.0, 0.0, 0.0), center=(0,0,0), color=(0.0, 1.0, 0.0)):
    """
    Generates vertices, indices, and colors for a transformed cube.

    Args:
        scale (tuple): The scaling factor for the cube along x, y, z.
        rotation_angles (tuple): Euler angles (in degrees) for rotation around x, y, z axes.
        center (tuple): The center position of the cube.
        color (tuple): The (R, G, B) color of the cube.

    Returns:
        tuple: A tuple containing (vertices, indices, colors) as numpy arrays.
    """
    # 1. Convert Euler angles (degrees) to a rotation matrix
    rx, ry, rz = np.radians(rotation_angles[0]), np.radians(rotation_angles[1]), np.radians(rotation_angles[2])
    
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    rot_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    
    rot_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    rot_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    # Combine rotations in Z-Y-X order
    rotation_matrix = rot_z @ rot_y @ rot_x
    
    # 2. Define unit cube vertices (centered at origin)
    base_vertices = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
    ], dtype=np.float32)

    # 3. Define cube face indices
    indices = np.array([
        0, 2, 1, 0, 3, 2,  # Front
        4, 5, 6, 4, 6, 7,  # Back
        4, 7, 3, 4, 3, 0,  # Left
        1, 6, 5, 1, 2, 6,  # Right
        5, 4, 0, 1, 5, 0,  # Bottom
        3, 6, 2, 3, 7, 6   # Top
    ], dtype=np.int32)

    # 4. Define vertex colors
    vert_colors = np.array([color for _ in range(8)], dtype=np.float32)

    # 5. Apply transformations
    scale_np = np.array(scale)
    center_np = np.array(center)
    
    transformed_vertices = base_vertices * scale_np
    transformed_vertices = transformed_vertices @ rotation_matrix.T
    transformed_vertices = transformed_vertices + center_np

    # 6. Edge of triangle indices

    edge_indices = np.array([
        # on edge
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        1,5, 2,6, 0,4, 3,7,
        # on face
        0,2, 4,6, 1,6, 3,4, 3,6, 0,5
    ], dtype=np.int32)




    rigid_2p_constraint_indices = np.array([
        # edge
        0, 1, 4, 5, 3, 2, 7, 6,
        1, 2, 5, 6, 0, 3, 4, 7,
        1, 5, 0, 4, 2, 6, 3, 7,
        # face diagonal
        1, 3, 0, 2, 5, 7, 4, 6,
        1, 6, 5, 2, 4, 3, 0, 7,
        2, 7, 6, 3, 1, 4, 5, 0,
        # body diagonal
        1, 7, 2, 4, 3, 5, 6, 0
    ],dtype=np.int32)

    scale_x, scale_y, scale_z = scale
    diag_xy = vec2f(scale_x, scale_y).norm()
    diag_yz = vec2f(scale_y, scale_z).norm()
    diag_xz = vec2f(scale_x, scale_z).norm()

    diag_xyz = vec2f(diag_xy, scale_z).norm()

    rigid_2p_constraint_distance = np.array([
        # edge
        scale_x, scale_x, scale_x, scale_x,
        scale_y, scale_y, scale_y, scale_y,
        scale_z, scale_z, scale_z, scale_z,
        # face diagonal
        diag_xy, diag_xy, diag_xy, diag_xy, 
        diag_yz, diag_yz, diag_yz, diag_yz, 
        diag_xz, diag_xz, diag_xz, diag_xz,
        # body diagonal
        diag_xyz, diag_xyz, diag_xyz, diag_xyz, 

    ],dtype=np.float32)

    return transformed_vertices.astype(np.float32), indices, vert_colors, rigid_2p_constraint_indices, rigid_2p_constraint_distance, edge_indices




import numpy as np
import trimesh
from typing import Dict, Tuple


import numpy as np
import trimesh
from typing import Dict, Tuple
import os # 引入 os 模块来检查文件

def load_mesh_for_simulation(
    obj_path: str,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
) -> Dict[str, np.ndarray]:
    # 1. 加载模型
    try:
        loaded_data = trimesh.load(obj_path, process=True, force='mesh')
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None

    if isinstance(loaded_data, trimesh.Scene):
        if len(loaded_data.geometry) == 0:
            raise ValueError("The loaded scene is empty and contains no mesh geometry.")
        mesh = trimesh.util.concatenate(list(loaded_data.geometry.values()))
    else:
        mesh = loaded_data

    # 新增: 将模型中心移至原点 (0,0,0)
    # 通过计算其包围盒的中心点，并将所有顶点进行相应的平移来实现。
    # 这可以确保后续的缩放、旋转和位移变换都是基于一个标准化的初始状态。
    mesh.apply_translation(-mesh.bounding_box.centroid)

    base_vertices = mesh.vertices
    base_normals = mesh.vertex_normals
    face_indices = mesh.faces

    # 2. 计算变换矩阵
    rx, ry, rz = np.radians(rotation_angles)
    rot_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    rot_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    rot_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    rotation_matrix_4x4 = trimesh.transformations.concatenate_matrices(rot_z, rot_y, rot_x)
    rotation_matrix_3x3 = rotation_matrix_4x4[:3, :3]

    scale_np = np.array(scale)
    center_np = np.array(center)

    # 3. 应用变换
    transformed_vertices = base_vertices * scale_np
    transformed_vertices = transformed_vertices @ rotation_matrix_3x3.T
    transformed_vertices = transformed_vertices + center_np

    transformed_normals = base_normals @ rotation_matrix_3x3.T
    
    # **【关键修改】**：修复法线归一化的维度问题
    # 计算范数时不保留维度，得到 (n,) 形状的数组
    norm_lengths = np.linalg.norm(transformed_normals, axis=1)
    
    # non_zero_len 现在是 (n,) 形状的布尔数组
    non_zero_len = norm_lengths > 1e-6
    
    # 在除法时，为了进行正确的广播 (n,3) / (n,1)，需要给除数增加一个维度
    # `norm_lengths[non_zero_len, np.newaxis]` 的形状是 (k, 1)，k 是非零法线的数量
    transformed_normals[non_zero_len] /= norm_lengths[non_zero_len, np.newaxis]

    # 4. 计算变换后的 PBD 约束
    constraint_indices = mesh.edges_unique
    v0_indices = constraint_indices[:, 0]
    v1_indices = constraint_indices[:, 1]
    p0 = transformed_vertices[v0_indices]
    p1 = transformed_vertices[v1_indices]
    constraint_distances = np.linalg.norm(p1 - p0, axis=1)

    # 4.5. 计算弯曲约束（二面角）
    constraint_bend_indices = np.array([], dtype=np.int32)
    constraint_bend_init_angle = np.array([], dtype=np.float32)

    # 仅当模型中存在面时才计算弯曲约束
    if face_indices.shape[0] > 0:
        # 使用变换后的顶点重新创建一个 Trimesh 对象以利用其拓扑功能
        # process=False 因为我们只关心拓扑信息，避免不必要的处理
        transformed_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=face_indices, process=False)
        
        # 检查是否存在相邻面
        if len(transformed_mesh.face_adjacency) > 0:
            # 获取共享边、二面角以及相邻面索引
            shared_edges = transformed_mesh.face_adjacency_edges
            init_angles_rad = transformed_mesh.face_adjacency_angles
            adjacent_face_pairs = transformed_mesh.face_adjacency

            p2_indices = []
            p3_indices = []

            # 对每个共享边，找到构成两个相邻三角形的另外两个顶点
            for i in range(len(adjacent_face_pairs)):
                face1_v_indices = face_indices[adjacent_face_pairs[i, 0]]
                face2_v_indices = face_indices[adjacent_face_pairs[i, 1]]
                shared_edge_v_indices = shared_edges[i]
                
                # 通过集合运算找到每个面中的独有顶点
                p2 = list(set(face1_v_indices) - set(shared_edge_v_indices))[0]
                p3 = list(set(face2_v_indices) - set(shared_edge_v_indices))[0]
                p2_indices.append(p2)
                p3_indices.append(p3)

            # 组合成 (n, 4) 的顶点索引数组
            # 顺序: 共享边的2个顶点, 第1个面的独立顶点, 第2个面的独立顶点
            p2_indices_np = np.array(p2_indices, dtype=np.int32).reshape(-1, 1)
            p3_indices_np = np.array(p3_indices, dtype=np.int32).reshape(-1, 1)
            bend_indices_quads = np.hstack([shared_edges, p2_indices_np, p3_indices_np])
            
            # 展平为 PBD 需要的一维格式
            constraint_bend_indices = bend_indices_quads.flatten()
            
            # init_angles_rad from trimesh is the angle between face normals.
            # The dihedral angle is pi minus this value.
            constraint_bend_init_angle = np.pi - init_angles_rad
            constraint_bend_init_angle[np.abs(constraint_bend_init_angle) < 1e-7] = 0.0
            constraint_bend_init_angle[np.abs(constraint_bend_init_angle - np.pi) < 1e-7] = np.pi


    # 5. 准备其他返回数据
    num_vertices = len(transformed_vertices)
    vertex_colors = np.tile(np.array(color), (num_vertices, 1))
    indices_flat = face_indices.flatten()

    # 6. 组合并返回结果
    
    result = {
        'vertices': transformed_vertices.astype(np.float32),
        'normals': transformed_normals.astype(np.float32),
        'colors': vertex_colors.astype(np.float32),
        'indices': indices_flat.astype(np.int32),
        'constraint_rigid_indices': constraint_indices.flatten().astype(np.int32),
        'constraint_rigid_distances': constraint_distances.astype(np.float32),
        'constraint_bend_indices': constraint_bend_indices.astype(np.int32),
        'constraint_bend_init_angle': constraint_bend_init_angle.astype(np.float32)
    }
    
    return result




#####################
# 1 -> segment hit triangle
# 0 -> segment do not hit, but the ray of segment hit
# -1 -> ray do not hit
#####################
@ti.func
def triangle_intersection_test(segment, triangle, extend=0.0):
    triangle_v1, triangle_v2, triangle_v3 = triangle
    segment_v1, segment_v2 = segment

    triangle_s1 = triangle_v2 - triangle_v1
    triangle_s2 = triangle_v3 - triangle_v1
    segment_s1 = segment_v2 - segment_v1

    D = segment_s1
    D_norm = D.norm()
    E1 = triangle_s1
    E2 = triangle_s2
    T = segment_v1 - triangle_v1

    P = tm.cross(D, E2)
    Q = tm.cross(T, E1)
    det = tm.dot(P, E1)

    n = tm.normalize(tm.cross(E1, E2))
    segment_v2_to_tri_vertical_distance = abs(tm.dot(T, n))
    
    result = 1
    facing = 0 # 1: front, -1: back, 0: parallel

    # Epsilon for float comparisons
    EPS = 1e-6

    # if det is close to 0, ray lies in plane of triangle
    if abs(det) < EPS:
        result = -1
    else:
        if det > 0:
            facing = 1
        else:
            facing = -1
        
        det_inv = 1 / det
        t = tm.dot(Q, E2) * det_inv
        u = tm.dot(P, T) * det_inv
        v = tm.dot(Q, D) * det_inv


        if t <= 0 or u <= 0 or v <= 0 or (u + v) > 1 - EPS:
            result = -1
        elif (t - 1) * D_norm - extend > 0:
            result = 0
        
    return ti.i32(result), ti.f32(segment_v2_to_tri_vertical_distance), ti.i32(facing)

# ti.init(arch=ti.vulkan)
# ti.init(debug=True)

# @ti.kernel
# def test():

#     a, b, c = triangle_intersection_test(
#         (vec3f(0,2,0), vec3f(0,0.6,0)), 
#         (vec3f(-3,0,0), vec3f(3,0,3), vec3f(3,0,-3)),
#         0.5
#         )
#     print(a)
#     print(b)
#     print(c)

# test()

@ti.func
def calculate_sin_abs_from_cos(cos_theta: ti.f32):
    cos2 = cos_theta * cos_theta
    sin2 = 1.0 - cos2
    return ti.sqrt(max(0.0, sin2))
