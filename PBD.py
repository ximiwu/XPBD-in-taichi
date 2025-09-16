from msilib import Directory
from pyexpat import model
import taichi as ti
import taichi.math as tm
from taichi.linalg import SparseMatrixBuilder, SparseSolver
from trimesh.proximity import thickness
from utils import *
import numpy as np
import imageio
import argparse

ti.init(arch=ti.vulkan)
ti.init(debug=True)

#TODO:
# better stiffness(k) implement
# velocity damp
# moving face moving point ccd
# 对于预测线段完全在物体内的情况，代码的实现是生成一个距离线段起点最近的面的4p碰撞constraint，与论文不同





@ti.dataclass
class ConstraintRigid_2P:
    indices: vec2i
    stiffness: ti.f32
    distance: ti.f32

CONSTRAINTRIGID_2P_MAX_NUM = 100000
constraints_rigid_2p = ConstraintRigid_2P.field(shape=CONSTRAINTRIGID_2P_MAX_NUM)
num_constraints_rigid_2p = ti.field(dtype=ti.i32, shape=())
num_constraints_rigid_2p[None] = 0

def add_rigid_2p_constraint(indices : vec2i, stiffness: ti.f32, distance: ti.f32):
    current_index = num_constraints_rigid_2p[None]
    if current_index >= CONSTRAINTRIGID_2P_MAX_NUM:
        print(f"error: add_rigid_2p_constraint")
        return

    constraints_rigid_2p[current_index].indices = indices
    constraints_rigid_2p[current_index].stiffness = stiffness
    constraints_rigid_2p[current_index].distance = distance
    num_constraints_rigid_2p[None] += 1


@ti.dataclass
class ConstraintBend_4P:
    indices: vec4i
    stiffness: ti.f32
    init_angle: ti.f32 # radiance, not degree

CONSTRAINTBEND_4P_MAX_NUM = 100000
constraints_bend_4p = ConstraintBend_4P.field(shape=CONSTRAINTBEND_4P_MAX_NUM)
num_constraints_bend_4p = ti.field(dtype=ti.i32, shape=())
num_constraints_bend_4p[None] = 0

def add_bend_4p_constraint(indices : vec4i, stiffness: ti.f32, init_angle: ti.f32):
    current_index = num_constraints_bend_4p[None]
    if current_index >= CONSTRAINTBEND_4P_MAX_NUM:
        print(f"error: add_bend_4p_constraint")
        return

    constraints_bend_4p[current_index].indices = indices
    constraints_bend_4p[current_index].stiffness = stiffness
    constraints_bend_4p[current_index].init_angle = init_angle
    num_constraints_bend_4p[None] += 1



@ti.dataclass
class ConstraintCollision_4P:
    point_index: ti.i32
    triangle_index: vec3i

CONSTRAINTCOLLISION_4P_MAX_NUM = 100000
constraints_collision_4p = ConstraintCollision_4P.field(shape=CONSTRAINTCOLLISION_4P_MAX_NUM)
num_constraints_collision_4p = ti.field(dtype=ti.i32, shape=())

@ti.func
def add_collision_4p_constraint(point_index : ti.i32, triangle_index : vec3i):

    current_index = num_constraints_collision_4p[None]
    if current_index >= CONSTRAINTCOLLISION_4P_MAX_NUM:
        print(f"error: add_collision_4p_constraint")
        current_index = 0

    constraints_collision_4p[current_index].point_index = point_index
    constraints_collision_4p[current_index].triangle_index = triangle_index
    num_constraints_collision_4p[None] += 1


@ti.dataclass
class ConstraintCollision_4P_DirectionThick:
    point_index: ti.i32
    triangle_index: vec3i
    thickness: ti.f32

CONSTRAINTCOLLISION_4P_DIRECTIONTHICK_MAX_NUM = 100000
constraints_collision_4p_direction_thick = ConstraintCollision_4P_DirectionThick.field(shape=CONSTRAINTCOLLISION_4P_DIRECTIONTHICK_MAX_NUM)
num_constraints_collision_4p_direction_thick = ti.field(dtype=ti.i32, shape=())

@ti.func
def add_collision_4p_direction_thick_constraint(point_index : ti.i32, triangle_index : vec3i, thickness: ti.f32):

    current_index = num_constraints_collision_4p_direction_thick[None]
    if current_index >= CONSTRAINTCOLLISION_4P_DIRECTIONTHICK_MAX_NUM:
        print(f"error: add_collision_4p_direction_thick_constraint")
        current_index = 0

    constraints_collision_4p_direction_thick[current_index].point_index = point_index
    constraints_collision_4p_direction_thick[current_index].triangle_index = triangle_index
    constraints_collision_4p_direction_thick[current_index].thickness = thickness
    num_constraints_collision_4p_direction_thick[None] += 1



all_verts_list = []
all_indices_list = []
all_colors_list = []
all_mass_inv_list = []
all_mass_list = []
all_vertices_thickness_list = []
all_face_span_list = []
all_idx_to_obj_list = []
all_edge_indices_list = []
vert_offset = 0
tri_offset = 0
num_object = 0

def add_obj_to_world(
    verts, inds, cols, mass, thickness=0.0, stiffness=0.5, 
    rigid_2p_constraint_indices=None, rigid_2p_constraint_distance=None, 
    bend_4p_constraint_indices=None, bend_4p_constraint_init_angle=None,
    edge_indices=None,
    ):
    global vert_offset, tri_offset, num_object
    all_verts_list.append(verts)
    all_indices_list.append(inds + vert_offset)
    all_colors_list.append(cols)
    if edge_indices is not None:
        all_edge_indices_list.append(edge_indices + vert_offset)

    current_num_tri = len(inds) // 3
    all_face_span_list.append(np.array([[tri_offset, tri_offset + current_num_tri - 1]],dtype=np.int32))
    all_idx_to_obj_list.append(np.full(len(verts), num_object, dtype=np.float32))
    if mass == -1:
        all_mass_inv_list.append(np.full(len(verts), 0, dtype=np.float32))
        all_mass_list.append(np.full(len(verts), -1, dtype=np.float32))
    else:
        all_mass_inv_list.append(1.0 / np.full(len(verts), mass, dtype=np.float32))
        all_mass_list.append(np.full(len(verts), mass, dtype=np.float32))

    all_vertices_thickness_list.append(np.full(len(verts), thickness, dtype=np.float32))
    
    if rigid_2p_constraint_indices is not None:
        for idx, distance in enumerate(rigid_2p_constraint_distance):
            add_rigid_2p_constraint(
                indices=vec2i(rigid_2p_constraint_indices[idx * 2 + 0] + vert_offset, rigid_2p_constraint_indices[idx * 2 + 1] + vert_offset),
                stiffness=stiffness,
                distance=distance
            )
    if bend_4p_constraint_indices is not None:
        for idx, angle in enumerate(bend_4p_constraint_init_angle):
            add_bend_4p_constraint(
                indices=vec4i(bend_4p_constraint_indices[idx * 4 + 0] + vert_offset, bend_4p_constraint_indices[idx * 4 + 1] + vert_offset, bend_4p_constraint_indices[idx * 4 + 2] + vert_offset, bend_4p_constraint_indices[idx * 4 + 3] + vert_offset, ),
                stiffness=stiffness,
                init_angle=angle
            )

    vert_offset += len(verts)
    tri_offset += current_num_tri
    num_object += 1


def add_cube(scale=(1.0, 1.0, 1.0), rotation_angles=(0.0, 0.0, 0.0), center=(0,0,0), color=(0.0, 1.0, 0.0), mass=1.0, stiffness=1, thickness=0.0):
    verts, inds, cols, rigid_2p_constraint_indices, rigid_2p_constraint_distance, edge_indices = gen_cube(scale, rotation_angles, center, color)
    add_obj_to_world(verts, inds, cols, mass, thickness, stiffness, rigid_2p_constraint_indices, rigid_2p_constraint_distance, edge_indices=edge_indices)


def load_mesh(
    obj_path: str,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_angles: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    mass=1.0, stiffness=1, thickness=0.0
):

    model_data = load_mesh_for_simulation(
        obj_path=obj_path,
        scale=scale,
        rotation_angles=rotation_angles,
        center=center,
        color=color
    )
    if model_data:
        add_obj_to_world(
            model_data['vertices'], model_data['indices'], model_data['colors'], mass, thickness, stiffness, 
            model_data['constraint_rigid_indices'], model_data['constraint_rigid_distances'], 
            model_data['constraint_bend_indices'], model_data['constraint_bend_init_angle'],
            )
    else:
        print("模型数据加载失败！")


def setup_scene():
    global vert_offset
    """
    Sets up the scene by generating multiple cubes.
    """
    add_cube(
        center=(0, 4.5, 0),
        scale=(0.5, 0.5, 0.5),
        rotation_angles=(0, 40, 0),
        color=(0.0, 1.0, 0.0),
        mass=10
    )
    # add_cube(
    #     center=(1, 7.5, 0),
    #     scale=(0.5, 0.5, 0.5),
    #     rotation_angles=(30, 20, 60),
    #     color=(0.0, 1.0, 0.0),
    #     mass=1
    # )
    # add_cube(
    #     center=(4, 30, 2),
    #     scale=(6, 2, 6),
    #     rotation_angles=(50, 0, 70),
    #     color=(1.0, 1.0, 0.0),
    #     mass=10
    # )
    # add_cube(
    #     center=(1, 10.5, 1),
    #     scale=(1.0, 1.0, 1.0),
    #     rotation_angles=(10, 50, 00),
    #     color=(0.5, 0.5, 0.5),
    #     mass=1
    # )
    # add_cube(
    #     center=(2, 11.5, 1),
    #     scale=(0.8, 0.4, 0.8),
    #     rotation_angles=(10, 70, 00),
    #     color=(0.5, 0.5, 0.5),
    #     mass=1
    # )


    # all_verts_list.append(np.array([[-1.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    # all_indices_list.append(np.array([0, 1, 2], dtype= np.int32) + vert_offset)
    # all_colors_list.append(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32))
    # all_mass_inv_list.append(np.array([0,0,0], dtype=np.float32))
    # vert_offset += 3



    add_cube(
        center=(0, -0.5, 0),
        scale=(100.0, 1, 100.0),
        color=(0.5, 0.5, 0.5),
        mass=-1
    )

    add_cube(
        center=(0, 0, -1),
        scale=(3, 6, 3),
        rotation_angles=(0, 40, 0),
        color=(1.0, 0.0, 0.0),
        mass=-1
    )

    load_mesh(
        obj_path='./PBD/reproduction/models/plane_30x30.obj',
        scale=(5.0, 5.0, 5.0),
        rotation_angles=(0, 45, 0),
        center=(0.0, 6.5, 0.0),
        color=(0.0, 0.0, 1.0),
        thickness= 0.1
    )




    # Convert lists to single numpy arrays
    final_verts = np.concatenate(all_verts_list, axis=0).astype(np.float32)
    final_indices = np.concatenate(all_indices_list, axis=0).astype(np.int32)
    final_colors = np.concatenate(all_colors_list, axis=0).astype(np.float32)
    final_inv_mass = np.concatenate(all_mass_inv_list, axis=0).astype(np.float32)
    final_mass = np.concatenate(all_mass_list, axis=0).astype(np.float32)
    final_vertices_thickness = np.concatenate(all_vertices_thickness_list, axis=0).astype(np.float32)
    final_obj_face_span = np.concatenate(all_face_span_list, axis=0).astype(np.int32)
    final_idx_to_obj = np.concatenate(all_idx_to_obj_list, axis=0).astype(np.int32)
    final_edge_indices = np.concatenate(all_edge_indices_list, axis=0).astype(np.int32)

    return final_verts, final_indices, final_colors, final_inv_mass, final_mass, final_vertices_thickness, final_obj_face_span, final_idx_to_obj, final_edge_indices

# Generate mesh data from numpy
verts_np, inds_np, colors_np, mass_inv_np, mass_np, vertices_thickness_np, obj_face_span_np, idx_to_obj_np, edge_indices_np = setup_scene()
num_vertices = verts_np.shape[0]
num_triangles = inds_np.shape[0] // 3
num_edge = edge_indices_np.shape[0] // 2



# Create Taichi fields to hold the data
vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
vertices_predict = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
indices = ti.field(dtype=ti.i32, shape=num_triangles * 3)
colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
mass_inv = ti.field(dtype=ti.f32, shape=num_vertices)
mass = ti.field(dtype=ti.f32, shape=num_vertices)
vertices_thickness = ti.field(dtype=ti.f32, shape=num_vertices)
obj_face_span = ti.field(dtype=vec2i, shape=num_object)
idx_to_obj = ti.field(dtype=ti.i32, shape=num_vertices)
edge_indices = ti.field(dtype=ti.i32, shape=num_edge * 2)

@ti.kernel
def initialize_data(
    verts_np: ti.types.ndarray(ti.f32, ndim=2),
    inds_np: ti.types.ndarray(ti.i32, ndim=1),
    colors_np: ti.types.ndarray(ti.f32, ndim=2),
    mass_inv_np : ti.types.ndarray(ti.f32, ndim=1),
    mass_np : ti.types.ndarray(ti.f32, ndim=1),
    vertices_thickness_np : ti.types.ndarray(ti.f32, ndim=1),
    obj_face_span_np: ti.types.ndarray(ti.i32, ndim=2),
    idx_to_obj_np: ti.types.ndarray(ti.i32, ndim=1),
    edge_indices_np: ti.types.ndarray(ti.i32, ndim=1)
):
    """Copy data from NumPy arrays to Taichi fields."""
    for i in range(num_vertices):
        vertices[i] = tm.vec3(verts_np[i, 0], verts_np[i, 1], verts_np[i, 2])
        colors[i] = tm.vec3(colors_np[i, 0], colors_np[i, 1], colors_np[i, 2])
        velocities[i].fill(0)
        mass_inv[i] = mass_inv_np[i]
        mass[i] = mass_np[i]
        vertices_thickness[i] = vertices_thickness_np[i]
        idx_to_obj[i] = idx_to_obj_np[i]

        # if i < 500:
        #     mass_inv[i] = 0.0

    for i in range(num_triangles * 3):
        indices[i] = inds_np[i]

    for i in range(num_object):
        obj_face_span[i] = vec2i(obj_face_span_np[i, 0], obj_face_span_np[i, 1])
    
    for i in range(num_edge * 2):
        edge_indices[i] = edge_indices_np[i]
    
    




# Initialize Taichi fields with the generated data
initialize_data(verts_np, inds_np, colors_np, mass_inv_np, mass_np, vertices_thickness_np, obj_face_span_np, idx_to_obj_np, edge_indices_np)


# ---------------------------------------------------------------------------- #
# Simulation
# ---------------------------------------------------------------------------- #

delta_t = 0.005
# delta_t = 0.01
solver_iteration_times = 10
gravity = vec3f(0, -9.8, 0)

@ti.func
def velocities_prediction():
    for i in range(num_vertices):
        if mass_inv[i] > 0.0:
            velocities[i] = velocities[i] + delta_t * gravity


obj_x_mean = ti.Vector.field(3, dtype=ti.f32, shape=num_object)
obj_v_mean = ti.Vector.field(3, dtype=ti.f32, shape=num_object)
obj_total_mass = ti.Vector.field(1, dtype=ti.f32, shape=num_object)
obj_vertices_num = ti.Vector.field(1, dtype=ti.i32, shape=num_object)

@ti.func
def velocities_damp():
    pass

    for current_obj in range(num_object):
        obj_x_mean[current_obj] = vec3f(0.0,0.0,0.0)
        obj_v_mean[current_obj] = vec3f(0.0,0.0,0.0)
        obj_vertices_num[current_obj] = 0
        obj_total_mass[current_obj] = 0.0
    ti.sync()
    
    for i in range(num_vertices):
        current_obj = idx_to_obj[i]
        vertex_mass = mass[i]
        ti.atomic_add(obj_x_mean[current_obj], vertices[i] * vertex_mass)
        ti.atomic_add(obj_v_mean[current_obj], velocities[i] * vertex_mass)
        ti.atomic_add(obj_vertices_num[current_obj], 1)
        ti.atomic_add(obj_total_mass[current_obj], vertex_mass)
    ti.sync()

    for current_obj in range(num_object):
        total_mass = obj_total_mass[current_obj]
        obj_x_mean[current_obj] = obj_x_mean[current_obj] / total_mass
        obj_v_mean[current_obj] = obj_v_mean[current_obj] / total_mass
    ti.sync()





@ti.func
def vertices_prediction():
    for i in range(num_vertices):
        if mass_inv[i] > 0.0:
            vertices_predict[i] = vertices[i] + delta_t * velocities[i]
        else:
            vertices_predict[i] = vertices[i] # Static objects don't move




@ti.func
def gen_collision_constraints():

    num_constraints_collision_4p[None] = 0
    num_constraints_collision_4p_direction_thick[None] = 0
    
    for i in range(num_vertices):

        if mass_inv[i] == 0:
            continue
        segment=(vertices[i], vertices_predict[i])

        current_obj = 0
        current_obj_ray_intersect_times = 0
        current_obj_nearest_face_idx = -1
        current_obj_nearest_face_vertical_distance = -1.0
        
        current_obj_face_span_end = obj_face_span[current_obj].y

        obj_idx = idx_to_obj[i]
        point_thickness = vertices_thickness[i]

        ti.loop_config(serialize=True)
        for j in range(num_triangles):
            
            idx0 = indices[j * 3 + 0]
            idx1 = indices[j * 3 + 1]
            idx2 = indices[j * 3 + 2]

            # if (obj_idx != current_obj):
            if (i != idx0 and i != idx1 and i != idx2):
                triangle = (vertices[idx0], vertices[idx1], vertices[idx2])
                average_thickness = point_thickness + ((vertices_thickness[idx0] + vertices_thickness[idx1] + vertices_thickness[idx2]) / 3.0)
                result, vertical_distance, facing = triangle_intersection_test(segment=segment, triangle=triangle, extend=point_thickness)                    

                if(current_obj_nearest_face_vertical_distance > vertical_distance or current_obj_nearest_face_vertical_distance == -1):
                    current_obj_nearest_face_vertical_distance = vertical_distance
                    current_obj_nearest_face_idx = j

                if(result == 1):
                    if (facing == 1):
                        if(average_thickness == 0.0):
                            add_collision_4p_constraint(point_index=i, triangle_index=vec3i(idx0, idx1, idx2))
                        else:
                            add_collision_4p_direction_thick_constraint(point_index=i, triangle_index=vec3i(idx0, idx1, idx2), thickness=average_thickness)
                    else:
                        if(average_thickness == 0.0):
                            add_collision_4p_constraint(point_index=i, triangle_index=vec3i(idx2, idx1, idx0))
                        else:
                            add_collision_4p_direction_thick_constraint(point_index=i, triangle_index=vec3i(idx2, idx1, idx0), thickness=average_thickness)
                    current_obj_ray_intersect_times += 1

                elif(result == 0):
                    current_obj_ray_intersect_times += 1

            if (j == current_obj_face_span_end):
                if((current_obj_ray_intersect_times % 2) != 0) and (idx_to_obj[i] != current_obj):
                    add_collision_4p_constraint(point_index=i, triangle_index=vec3i(indices[current_obj_nearest_face_idx * 3 + 0], indices[current_obj_nearest_face_idx * 3 + 1], indices[current_obj_nearest_face_idx * 3 + 2]))
                current_obj += 1
                if current_obj < num_object:
                    current_obj_face_span_end = obj_face_span[current_obj].y
                current_obj_ray_intersect_times = 0
                current_obj_nearest_face_idx = -1
                current_obj_nearest_face_vertical_distance = -1

                




@ti.kernel
def simulation_prediction():
    velocities_prediction()
    velocities_damp()
    vertices_prediction()
    gen_collision_constraints()




# 传入的三角面顶点顺序按照正面的右手法则（逆时针）。
# 对于不封闭的mesh，三角面背面碰撞则传入的顶点顺序按照背面的右手法则（逆时针），也就是正面的顺时针
# 因为constraint只需考虑自己控制的四个点的相对关系，而不考虑整体mesh
@ti.kernel
def simulation_solving():
    ti.loop_config(serialize=True)
    for _ in range(solver_iteration_times):



        # constraints_rigid_2p
        ti.loop_config(serialize=True)
        for i in range(num_constraints_rigid_2p[None]):
            idx1, idx2 = constraints_rigid_2p[i].indices.xy
            stiffness = constraints_rigid_2p[i].stiffness
            distance = constraints_rigid_2p[i].distance

            p1 = vertices_predict[idx1]
            p2 = vertices_predict[idx2]
            w1 = mass_inv[idx1]
            w2 = mass_inv[idx2]
            
            if w1 + w2 < 1e-9:
                continue
            
            w1_add_w2_inv = 1.0 / (w1 + w2)

            current_distance_vec = p1 - p2
            current_distance_norm = tm.length(current_distance_vec)

            if current_distance_norm < 1e-9:
                continue

            segment_direction = current_distance_vec / current_distance_norm
            delta_distance = current_distance_norm - distance

            delta_p1 = -(w1 * w1_add_w2_inv) * delta_distance * segment_direction
            delta_p2 = +(w2 * w1_add_w2_inv) * delta_distance * segment_direction

            vertices_predict[idx1] += delta_p1 * 0.8
            vertices_predict[idx2] += delta_p2 * 0.8


        # constraints_bend_4p
        ti.loop_config(serialize=True)
        for i in range(num_constraints_bend_4p[None]):
            stiffness = constraints_bend_4p[i].stiffness
            p_indices = constraints_bend_4p[i].indices
            init_angle = constraints_bend_4p[i].init_angle
            p1_idx, p2_idx, p3_idx, p4_idx = p_indices[0], p_indices[1], p_indices[2], p_indices[3]
            
            p1 = vertices_predict[p1_idx]
            p2 = vertices_predict[p2_idx] - p1
            p3 = vertices_predict[p3_idx] - p1
            p4 = vertices_predict[p4_idx] - p1

            w1 = mass_inv[p1_idx]
            w2 = mass_inv[p2_idx]
            w3 = mass_inv[p3_idx]
            w4 = mass_inv[p4_idx]


            eps = 1e-6

            c23 = tm.cross(p2, p3)
            c24 = tm.cross(p2, p4)
            l23 = tm.max(c23.norm(), eps)
            l24 = tm.max(c24.norm(), eps)

            n1 = c23 / l23
            n2 = c24 / l24

            d = tm.dot(n1, n2)
            d = tm.clamp(d, -1.0 + eps, 1.0 - eps)
            theta = tm.acos(d)
            sin_theta = tm.sqrt(tm.max(1.0 - d * d, eps))

            
            q3 = (tm.cross(p2, n2) + tm.cross(n1, p2) * d) / l23
            q4 = (tm.cross(p2, n1) + tm.cross(n2, p2) * d) / l24
            q2 = -(tm.cross(p3, n2) + tm.cross(n1, p3) * d) / l23 \
                 - (tm.cross(p4, n1) + tm.cross(n2, p4) * d) / l24
            q1 = -q2 - q3 - q4


            den = w1 * tm.dot(q1, q1) + w2 * tm.dot(q2, q2) + w3 * tm.dot(q3, q3) + w4 * tm.dot(q4, q4)
            if (abs(den) < eps):
                continue

            C = theta - init_angle
            
            k = -4.0 * sin_theta * C / den


            delta_p1 = w1 * k * q1 * 0.01
            delta_p2 = w2 * k * q2 * 0.01
            delta_p3 = w3 * k * q3 * 0.01
            delta_p4 = w4 * k * q4 * 0.01

            vertices_predict[p1_idx] += delta_p1
            vertices_predict[p2_idx] += delta_p2
            vertices_predict[p3_idx] += delta_p3
            vertices_predict[p4_idx] += delta_p4


        # constraints_collision_4p
        ti.loop_config(serialize=True)
        for i in range(num_constraints_collision_4p[None]):
            point_index = constraints_collision_4p[i].point_index
            tri_index_1, tri_index_2, tri_index_3 = constraints_collision_4p[i].triangle_index.xyz

            q = vertices_predict[point_index]
            p1 = vertices_predict[tri_index_1]
            p2 = vertices_predict[tri_index_2]
            p3 = vertices_predict[tri_index_3]

            p2p1 = p2 - p1
            p3p1 = p3 - p1
            qp1 = q - p1

            dC_dq = tm.cross(p2p1, p3p1)

            C = tm.dot(qp1, dC_dq)
            if C >= 0.0:
                continue

            w_q = mass_inv[point_index]
            w_p1 = mass_inv[tri_index_1]
            w_p2 = mass_inv[tri_index_2]
            w_p3 = mass_inv[tri_index_3]


            
            
            dC_dp2 = tm.cross(p3p1, qp1)
            dC_dp3 = tm.cross(qp1, p2p1)
            dC_dp1 = -(dC_dq + dC_dp2 + dC_dp3)

            s = C / (w_q * tm.dot(dC_dq, dC_dq) + w_p1 * tm.dot(dC_dp1, dC_dp1) + w_p2 * tm.dot(dC_dp2, dC_dp2) + w_p3 * tm.dot(dC_dp3, dC_dp3) + 0.000001)

            vertices_predict[point_index] += (-s * w_q * dC_dq)
            vertices_predict[tri_index_1] += (-s * w_p1 * dC_dp1)
            vertices_predict[tri_index_2] += (-s * w_p2 * dC_dp2)
            vertices_predict[tri_index_3] += (-s * w_p3 * dC_dp3)        

        # constraints_collision_4p_direction_thick
        ti.loop_config(serialize=True)
        for i in range(num_constraints_collision_4p_direction_thick[None]):
            point_index = constraints_collision_4p_direction_thick[i].point_index
            tri_index_1, tri_index_2, tri_index_3 = constraints_collision_4p_direction_thick[i].triangle_index.xyz
            thickness = constraints_collision_4p_direction_thick[i].thickness

            h = thickness
            q = vertices_predict[point_index]
            p1 = vertices_predict[tri_index_1]
            p2 = vertices_predict[tri_index_2]
            p3 = vertices_predict[tri_index_3]

            p2p1 = p2 - p1
            p3p1 = p3 - p1
            qp1 = q - p1

            n = tm.cross(p2p1, p3p1)
            n_norm = n.norm()
            if n_norm < 1e-9:
                continue
            
            n_normalized = n / n_norm

            C = tm.dot(qp1, n_normalized) - h
            if C >= 0.0:
                continue

            w_q = mass_inv[point_index]
            w_p1 = mass_inv[tri_index_1]
            w_p2 = mass_inv[tri_index_2]
            w_p3 = mass_inv[tri_index_3]

            r_vertical = qp1 - (tm.dot(qp1, n_normalized) * n_normalized)
            
            dC_dq = n_normalized

            dC_dp2 = tm.cross(p3p1, r_vertical) / n_norm
            dC_dp3 = tm.cross(r_vertical, p2p1) / n_norm
            dC_dp1 = -(dC_dq + dC_dp2 + dC_dp3)

            s = C / (w_q * tm.dot(dC_dq, dC_dq) + w_p1 * tm.dot(dC_dp1, dC_dp1) + w_p2 * tm.dot(dC_dp2, dC_dp2) + w_p3 * tm.dot(dC_dp3, dC_dp3) + 0.000001)

            vertices_predict[point_index] += (-s * w_q * dC_dq)
            vertices_predict[tri_index_1] += (-s * w_p1 * dC_dp1)
            vertices_predict[tri_index_2] += (-s * w_p2 * dC_dp2)
            vertices_predict[tri_index_3] += (-s * w_p3 * dC_dp3)


            

            

@ti.func
def velocities_update():
    for i in range(num_vertices):
        velocities[i] = (vertices_predict[i] - vertices[i]) / delta_t

@ti.func
def vertices_update():
    for i in range(num_vertices):
        vertices[i] = vertices_predict[i]



@ti.func
def apply_collision_impulse_child_func(constraint_field, constraint_num, velocity_absorb):
    for i in range(num_constraints_collision_4p[None]):
        point_index = constraints_collision_4p[i].point_index
        tri_index_1, tri_index_2, tri_index_3 = constraints_collision_4p[i].triangle_index.xyz

        p1 = vertices[tri_index_1]
        p2 = vertices[tri_index_2]
        p3 = vertices[tri_index_3]

        p2p1 = p2 - p1
        p3p1 = p3 - p1

        normal = tm.normalize(tm.cross(p2p1, p3p1))
        if(velocity_absorb):
            velocities[point_index] = velocity_collision_absorb_friction(velocities[point_index], normal)
            velocities[tri_index_1] = velocity_collision_absorb_friction(velocities[tri_index_1], normal)
            velocities[tri_index_2] = velocity_collision_absorb_friction(velocities[tri_index_2], normal)
            velocities[tri_index_3] = velocity_collision_absorb_friction(velocities[tri_index_3], normal)
        else:
            velocities[point_index] = velocity_collision_reflection_friction(velocities[point_index], normal)
            velocities[tri_index_1] = velocity_collision_reflection_friction(velocities[tri_index_1], normal)
            velocities[tri_index_2] = velocity_collision_reflection_friction(velocities[tri_index_2], normal)
            velocities[tri_index_3] = velocity_collision_reflection_friction(velocities[tri_index_3], normal)

@ti.func
def apply_collision_impulse():
    apply_collision_impulse_child_func(constraints_collision_4p, num_constraints_collision_4p[None], False)
    apply_collision_impulse_child_func(constraints_collision_4p_direction_thick, num_constraints_collision_4p_direction_thick[None], False)



@ti.func
def velocity_collision_reflection_friction(velocity : vec3f, normal : vec3f):
    normal_term = tm.dot(velocity, normal) * normal
    return (velocity - normal_term) * 0.9 - normal_term

@ti.func
def velocity_collision_absorb_friction(velocity : vec3f, normal : vec3f):
    normal_term = tm.dot(velocity, normal) * normal
    return (velocity - normal_term) * 0.9
        

@ti.kernel
def simulation_update():
    velocities_update()
    vertices_update()
    apply_collision_impulse()




# ---------------------------------------------------------------------------- #
# Argument Parser
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="PBD Simulation with Taichi")
parser.add_argument('--record', type=str, help="Record simulation to a video file (e.g., output.mp4)")
parser.add_argument('--fps', type=int, default=30, help="Frame rate for the recorded video")
args = parser.parse_args()


# ---------------------------------------------------------------------------- #
# GUI
# ---------------------------------------------------------------------------- #
width, height = 1280, 720
window = ti.ui.Window("PBD Cube Simulation", (width, height), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.1, 0.1, 0.2))
scene = window.get_scene()


camera = ti.ui.Camera()
camera.position(9, 2.5, -9)
camera.lookat(3, 0.5, 3)
camera.up(0, 1, 0)
camera.projection_mode(ti.ui.ProjectionMode.Perspective)


refline = ti.Vector.field(3, dtype=ti.f32, shape=6)

refline[0] = vec3f(-100, 0.01, 0)
refline[1] = vec3f(100, 0.01, 0)
refline[2] = vec3f(0, -100, 0)
refline[3] = vec3f(0, 100, 0)
refline[4] = vec3f(0, 0.01, -100)
refline[5] = vec3f(0, 0.01, 100)



# ---------------------------------------------------------------------------- #
# Wait for start
# ---------------------------------------------------------------------------- #
# program_started = False
# while window.running and not program_started:

#     for e in window.get_events(ti.ui.PRESS):
#         if e.key == ti.ui.ESCAPE:
#             window.running = False
#         elif e.key == ti.ui.SPACE:
#             program_started = True

# ---------------------------------------------------------------------------- #
# Video Recorder
# ---------------------------------------------------------------------------- #
video_writer = None
if args.record:
    print(f"Recording video to {args.record} at {args.fps} FPS")
    video_writer = imageio.get_writer(args.record, fps=args.fps)


# ---------------------------------------------------------------------------- #
# Runtime Loop
# ---------------------------------------------------------------------------- #


while window.running:
    # --- Camera ---
    camera.track_user_inputs(window, movement_speed=0.1, yaw_speed=4.0, pitch_speed= 4.0, hold_key=ti.ui.LMB)
    scene.set_camera(camera)
    
    scene.ambient_light((0.2, 0.2, 0.2))
    scene.point_light(pos=(5, 10000000000, 5), color=(1, 1, 1))

    # --- Simulation Step ---
    simulation_prediction()
    simulation_solving()
    simulation_update()

    # --- Rendering ---
    scene.mesh(
        vertices,
        indices=indices,
        per_vertex_color=colors,
    )

    # scene.lines(vertices=vertices, width=2, indices=edge_indices, color=(0.15, 0.15, 0.15))
    # scene.lines(vertices=refline, width=1, color=(1, 1, 1))
    
    canvas.scene(scene)
    window.show()

    if video_writer is not None:
        img_float = window.get_image_buffer_as_numpy()
        img_transposed = np.transpose(img_float, (1, 0, 2))
        img_flipped = np.flip(img_transposed, axis=0) # Vertically flip the transposed image
        img_uint8 = (img_flipped * 255).astype(np.uint8)
        video_writer.append_data(img_uint8)


if video_writer is not None:
    video_writer.close()
    print("Video recording finished.")


