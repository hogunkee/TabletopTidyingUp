import copy
import os
import numpy as np
import nvisii as nv
import pybullet as p 
from transform_utils import euler2quat

def get_pybullet_init_euler():
    init_euler = {}
    init_euler[2] = [0, 0, 1]
    init_euler[4] = [0, 0, -1]
    init_euler[7] = [1, 0, 0]
    init_euler[8] = [1, 0, 0]
    init_euler[9] = [1, 0, 0]
    init_euler[10] = [1, 0, -1]
    init_euler[11] = [1, 0, -1]
    init_euler[12] = [0, 0, -1]
    init_euler[13] = [0, 0, 1]
    init_euler[14] = [0, 0, -1]
    init_euler[17] = [1, 0, -1]
    init_euler[20] = [0, 0, -1]
    init_euler[22] = [0, 0, -1]
    init_euler[24] = [0, 0, 1]
    init_euler[25] = [0, 0, -1]
    init_euler[26] = [0, 0, 2]
    init_euler[27] = [0, 0, 2]
    init_euler[28] = [0, 0, 2]
    init_euler[30] = [1, 0, -1]
    init_euler[31] = [1, 0, -1]
    init_euler[33] = [0, 0, 1]
    init_euler[36] = [0, 0, 2]
    init_euler[37] = [0, 0, 2]
    init_euler[38] = [1, 0, -1]
    init_euler[39] = [0, 0, 1]
    init_euler[40] = [0, 0, 2]
    init_euler[42] = [0, 0, -1]
    init_euler[43] = [0, 0, 1]
    init_euler[44] = [1, 0, -1]
    init_euler[46] = [0, 0, 2]
    init_euler[48] = [0, 0, -1]
    init_euler[54] = [0, 0, 1]
    init_euler[58] = [1, 0, 0]
    init_euler[60] = [0, 0, -1]
    init_euler[61] = [0, 0, -1]
    init_euler[65] = [0, 0, 2]
    init_euler[66] = [0, 0, 2]
    init_euler[67] = [0, 0, 2]
    init_euler[68] = [0, 0, 1]
    init_euler[69] = [0, 0, -1]
    init_euler[70] = [0, 0, -1]
    init_euler[71] = [0, 0, 2]
    init_euler[73] = [0, 0, 1]
    init_euler[74] = [0, -1, -1]
    init_euler[75] = [0, 0, 2]
    init_euler[79] = [0, 0, 2]
    init_euler[80] = [0, 0, 1]
    init_euler[81] = [0, 0, -1]
    init_euler[82] = [0, 0, 1]
    init_euler[83] = [0, 0, 1]
    init_euler[90] = [0, 0, 2]
    init_euler[93] = [0, 0, 1]
    return init_euler

def get_ycb_init_euler():
    init_euler = {}
    return init_euler

def get_init_euler():
    init_euler = {}
    pybullet_init_euler = get_pybullet_init_euler()
    ycb_init_euler = get_ycb_init_euler()
    for p in pybullet_init_euler:
        init_euler['pybullet-%d'%p] = pybullet_init_euler[p]
    for y in ycb_init_euler:
        init_euler['ycb-%d'%y] = ycb_init_euler[y]
    return init_euler

def generate_scene_random(num_objects):
    positions = []
    rotations = []
    for i in range(num_objects):
        pos, rot = get_random_pos_orn([0,0,0,1])
        positions.append(pos)
        rotations.append(rot)

    return positions, rotations

def generate_scene_shape(scene_type, num_objects):
    z_default = 0.7
    if scene_type=='random':
        positions = np.array(random_pos_on_table())
        # positions[:, 2] = z_default
        rotations = [] # TODO: add random rotations (z-axis)
    elif scene_type=='line':
        distance_enough = False
        while not distance_enough:
            x1, y1 = np.array(random_pos_on_table())[:2]
            x2, y2 = np.array(random_pos_on_table())[:2]

            distance_enough = (np.linalg.norm([x1 - x2, y1 - y2]) > 0.3) \
                                and (np.linalg.norm([x1 - x2, y1 - y2]) < 0.5)
        xs = np.linspace(x1, x2, num_objects)
        ys = np.linspace(y1, y2, num_objects)
        zs = np.ones(num_objects) * z_default + np.arange(num_objects) * 0.05
        positions = np.concatenate([xs, ys, zs]).reshape(3, num_objects).T
        rotations = []
    return positions, rotations

def generate_scene_at_center(scene_type, num_objects):
    z_default = 0.15
    if scene_type=='random':
        positions = 0.3*(np.random.rand(num_objects, 3) - 0.5)
        positions[:, 2] = z_default
    elif scene_type=='line':
        distance_enough = False
        while not distance_enough:
            x1, x2, y1, y2 = 0.5 * (np.random.rand(4) - 0.5)
            distance_enough = (np.linalg.norm([x1 - x2, y1 - y2]) > 0.3) \
                                and (np.linalg.norm([x1 - x2, y1 - y2]) < 0.5)
        xs = np.linspace(x1, x2, num_objects)
        ys = np.linspace(y1, y2, num_objects)
        zs = np.ones(num_objects) * z_default + np.arange(num_objects) * 0.05
        positions = np.concatenate([xs, ys, zs]).reshape(3, num_objects).T
    return positions

def get_rotation(roll, pitch, yaw):
    euler = roll, pitch, yaw
    x, y, z, w = euler2quat(euler)
    return x, y, z, w

def get_contact_objects():
    contact_pairs = set()
    for contact in p.getContactPoints():
        body_A = contact[1]
        body_B = contact[2]
        contact_pairs.add(tuple(sorted((body_A, body_B))))
    collisions = set()
    for cp in contact_pairs:
        if cp[0] == 0:
            continue
        collisions.add(cp)
    return collisions

def get_velocity(object_ids):
    velocities_linear = []
    velocities_rotation = []
    for pid in object_ids:
        vel_linear, vel_rot = p.getBaseVelocity(pid)
        velocities_linear.append(vel_linear)
        velocities_rotation.append(vel_rot)
    return velocities_linear, velocities_rotation

def update_visual_objects(object_ids, pkg_path, nv_objects=None):
    # object ids are in pybullet engine
    # pkg_path is for loading the object geometries
    # nv_objects refers to the already entities loaded, otherwise it is going 
    # to load the geometries and create entities. 
    if nv_objects is None:
        nv_objects = { }
    objs = copy.deepcopy(object_ids)
    objs.append(1)
    for object_id in objs:
        for idx, visual in enumerate(p.getVisualShapeData(object_id)):

            # Extract visual data from pybullet
            objectUniqueId = visual[0]
            linkIndex = visual[1]
            visualGeometryType = visual[2]
            dimensions = visual[3]
            meshAssetFileName = visual[4]
            local_visual_frame_position = visual[5]
            local_visual_frame_orientation = visual[6]
            rgbaColor = visual[7]

            if linkIndex == -1:
                dynamics_info = p.getDynamicsInfo(object_id,-1)
                inertial_frame_position = dynamics_info[3]
                inertial_frame_orientation = dynamics_info[4]
                base_state = p.getBasePositionAndOrientation(objectUniqueId)
                world_link_frame_position = base_state[0]
                world_link_frame_orientation = base_state[1]    
                m1 = nv.translate(nv.mat4(1), nv.vec3(inertial_frame_position[0], inertial_frame_position[1], inertial_frame_position[2]))
                m1 = m1 * nv.mat4_cast(nv.quat(inertial_frame_orientation[3], inertial_frame_orientation[0], inertial_frame_orientation[1], inertial_frame_orientation[2]))
                m2 = nv.translate(nv.mat4(1), nv.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
                m2 = m2 * nv.mat4_cast(nv.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))
                m = nv.inverse(m1) * m2
                q = nv.quat_cast(m)
                world_link_frame_position = m[3]
                world_link_frame_orientation = q
            else:
                linkState = p.getLinkState(objectUniqueId, linkIndex)
                world_link_frame_position = linkState[4]
                world_link_frame_orientation = linkState[5]
            
            # Name to use for components
            object_name = f"{objectUniqueId}_{linkIndex}_{idx}"

            meshAssetFileName = meshAssetFileName.decode('UTF-8')
            if object_name not in nv_objects:
                # Create mesh component if not yet made
                if visualGeometryType == p.GEOM_MESH:
                    try:
                        #print(meshAssetFileName)
                        nv_objects[object_name] = nv.import_scene(
                            os.path.join(pkg_path, meshAssetFileName)
                        )
                    except Exception as e:
                        print(e)
                elif visualGeometryType == p.GEOM_BOX:
                    assert len(meshAssetFileName) == 0
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_box(
                            name=object_name,
                            # half dim in nv.v.s. pybullet
                            size=nv.vec3(dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2)
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                elif visualGeometryType == p.GEOM_CYLINDER:
                    assert len(meshAssetFileName) == 0
                    length = dimensions[0]
                    radius = dimensions[1]
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_cylinder(
                            name=object_name,
                            radius=radius,
                            size=length / 2,    # size in nv.is half of the length in pybullet
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                elif visualGeometryType == p.GEOM_SPHERE:
                    assert len(meshAssetFileName) == 0
                    nv_objects[object_name] = nv.entity.create(
                        name=object_name,
                        mesh=nv.mesh.create_sphere(
                            name=object_name,
                            radius=dimensions[0],
                        ),
                        transform=nv.transform.create(object_name),
                        material=nv.material.create(object_name),
                    )
                else:
                    # other primitive shapes currently not supported
                    continue

            if object_name not in nv_objects: continue

            # Link transform
            m1 = nv.translate(nv.mat4(1), nv.vec3(world_link_frame_position[0], world_link_frame_position[1], world_link_frame_position[2]))
            m1 = m1 * nv.mat4_cast(nv.quat(world_link_frame_orientation[3], world_link_frame_orientation[0], world_link_frame_orientation[1], world_link_frame_orientation[2]))

            # Visual frame transform
            m2 = nv.translate(nv.mat4(1), nv.vec3(local_visual_frame_position[0], local_visual_frame_position[1], local_visual_frame_position[2]))
            m2 = m2 * nv.mat4_cast(nv.quat(local_visual_frame_orientation[3], local_visual_frame_orientation[0], local_visual_frame_orientation[1], local_visual_frame_orientation[2]))

            # import scene directly with mesh files
            if isinstance(nv_objects[object_name], nv.scene):
                # Set root transform of visual objects collection to above transform
                nv_objects[object_name].transforms[0].set_transform(m1 * m2)
                nv_objects[object_name].transforms[0].set_scale(dimensions)

                for m in nv_objects[object_name].materials:
                    m.set_base_color((rgbaColor[0] ** 2.2, rgbaColor[1] ** 2.2, rgbaColor[2] ** 2.2))
            # for entities created for primitive shapes
            else:
                nv_objects[object_name].get_transform().set_transform(m1 * m2)
                nv_objects[object_name].get_material().set_base_color(
                    (
                        rgbaColor[0] ** 2.2,
                        rgbaColor[1] ** 2.2,
                        rgbaColor[2] ** 2.2,
                    )
                )
            # print(visualGeometryType)
    return nv_objects

def remove_visual_objects(nv_objects):
    # pkg_path is for loading the object geometries
    for object_name in nv_objects:
        #entity = nv_objects[object_name].entities[0]
        for entity in nv_objects[object_name].entities:
            material = entity.get_material().get_name()
            mesh = entity.get_mesh().get_name()
            transform = entity.get_transform().get_name()

            nv.material.remove(material)
            nv.mesh.remove(mesh)
            nv.transform.remove(transform)

            entity.clear_material()
            entity.clear_mesh()
            entity.clear_transform()
            e_name = entity.get_name()
            nv.entity.remove(e_name)
    return None

def clear_scene():
    nv.camera.clear_all()
    nv.light.clear_all()
    nv.mesh.clear_all()
    nv.material.clear_all()
    nv.volume.clear_all()
    nv.transform.clear_all()
    nv.texture.clear_all()
    nv.entity.clear_all()
    nv.clear_all()
    return

def pickable_objects_list(objects_list, sg):
    pickable_objects = []
    for obj_id in objects_list.keys():
        pickable = True
        for pair in sg['on']:
            if obj_id == pair[0]:
                pickable = False
                break
        if pickable:
            pickable_objects.append(obj_id)
            
    return pickable_objects


def cal_distance(objects_list):
    dist = {}
    for i in objects_list.keys():
        for j in objects_list.keys():
            if i == j:
                continue
            else:
                dist_ = p.getClosestPoints(i, j, 2.0)
                dist[str((i,j))] = 1000
                for idx in range(len(dist_)):
                    dist[str((i,j))] = min(dist_[idx][8], dist[str((i,j))])
 
    return dist

def check_on_table(objects_list):
    print("check on table & image")
    for obj_id in objects_list.keys():
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        if(-0.4 < pos[0] < 0.4 and -0.6 < pos[1] < 0.6 and 0.6 < pos[2] < 1.0):
            pass
        else:
            print(objects_list[obj_id], "is not on the table")
            return False 
    return True           

def random_pos_on_table():
    pos_x = np.random.uniform(-0.3, 0.3)
    pos_y = np.random.uniform(-0.4, 0.4)
    pos_z = np.random.uniform(0.75, 0.8)
    return pos_x, pos_y, pos_z


def get_random_pos_orn(orig_orn):
    rot_angle = np.random.uniform(-180, 180) # counter-clockwise
    pos = random_pos_on_table()
    quat = p.getQuaternionFromEuler([0,0,rot_angle * np.pi / 180.0])
    new_rot = quaternion_multiply(quat, orig_orn)
    return pos, new_rot

def quaternion_multiply(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)
    
    
def move_object(obj_id, pos, orn):
    p.resetBasePositionAndOrientation(obj_id, pos, orn)
    for i in range(100):
        p.stepSimulation()