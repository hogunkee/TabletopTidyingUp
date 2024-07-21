import copy
import os
import numpy as np
import nvisii as nv
import pybullet as p 
from transform_utils import euler2quat

def get_object_categories():
    if os.path.exists('render_category/categories_dict.txt'):
        with open('render_category/categories_dict.txt', 'r') as f:
            lines = f.readlines()
        dict_str = ''.join([line.replace('\n','').replace('\t','') for line in lines])
        category_dic = eval(dict_str)
        return category_dic
    return None

def get_pybullet_init_euler():
    init_euler = {}
    if os.path.exists('euler_pybullet_new.csv'):
        with open('euler_pybullet_new-1.csv', 'r') as f:
            lines = f.readlines()
        for line in lines:
            e = line.replace('\n', '').split(',')
            init_euler[int(e[0])] = [float(e[1]), float(e[2]), float(e[3]), float(e[4])]
    return init_euler

def get_ycb_init_euler():
    init_euler = {}
    if os.path.exists('euler_ycb_new.csv'):
        with open('euler_ycb_new.csv', 'r') as f:
            lines = f.readlines()
        for line in lines:
            e = line.replace('\n', '').split(',')
            init_euler[int(e[0])] = [float(e[1]), float(e[2]), float(e[3]), float(e[4])]
    return init_euler

def get_housecat_init_euler():
    init_euler = {}
    if os.path.exists('euler_housecat_new.csv'):
        with open('euler_housecat_new.csv', 'r') as f:
            lines = f.readlines()
        for line in lines:
            e = line.replace('\n', '').split(',')
            init_euler[int(e[0])] = [float(e[1]), float(e[2]), float(e[3]), float(e[4])]
    for i in range(194):
        if i not in init_euler:
            init_euler[i] = [1, 0, 0, 1]
    return init_euler

def get_init_euler():
    init_euler = {}
    pybullet_init_euler = get_pybullet_init_euler()
    ycb_init_euler = get_ycb_init_euler()
    housecat_init_euler = get_housecat_init_euler()
    for p in pybullet_init_euler:
        init_euler['pybullet-%d'%p] = pybullet_init_euler[p]
    for y in ycb_init_euler:
        init_euler['ycb-%d'%y] = ycb_init_euler[y]
    for h in housecat_init_euler:
        init_euler['housecat-%d'%h] = housecat_init_euler[h]
    return init_euler

def get_random_pos_from_grid():
    x = np.linspace(-0.2, 0.3, 4)
    y = np.linspace(-0.4, 0.4, 6)
    z = np.linspace(0.8, 1.0, 2)
    pos = (np.random.choice(x), np.random.choice(y), np.random.choice(z))
    return pos

def generate_scene_random(num_objects):
    positions = []
    rotations = []
    for i in range(num_objects):
        _, rot = get_random_pos_orn([0,0,0,1])
        pos = get_random_pos_from_grid()
        
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
        num_trials = 0
        check_feasible = False
        while not check_feasible:
            xc, yc = np.array(random_pos_on_table())[:2]
            #x_hat = np.random.uniform(size=2) - 0.5
            x_hats = [[1, 0], [0, 1], [-1, 0], [0, -1],
                     [1, 1], [-1, 1], [-1, -1], [1, -1]]
            s = np.random.choice(range(8))
            x_hat = x_hats[s]
            x_hat /= np.linalg.norm(x_hat)
            x1, y1 = [xc, yc] - x_hat * 0.2 * num_objects/2 #0.15
            x2, y2 = [xc, yc] + x_hat * 0.2 * num_objects/2 #0.15
            if min(x1, x2) < -0.3 or max(x1, x2) > 0.3:
                check_feasible = False
                continue
            if min(y1, y2) < -0.4 or max(y1, y2) > 0.4:
                check_feasible = False
                continue
            check_feasible = True
            num_trials += 1
            if num_trials > 10:
                return None, None

        xs = np.linspace(x1, x2, num_objects)
        ys = np.linspace(y1, y2, num_objects)
        zs = np.ones(num_objects) * z_default + np.arange(num_objects) * 0.05
        positions = np.concatenate([xs, ys, zs]).reshape(3, num_objects).T
        rotations = []
    elif scene_type=='line-rotated':
        check_feasible = False
        while not check_feasible:
            interval = np.random.uniform(0.15, 0.25)
            rot90 = np.random.choice(2)

            xc, yc = np.array(random_pos_on_table())[:2]
            #x_hat = np.random.uniform(size=2) - 0.5
            x_hats = [[1, 0], [0, 1], [-1, 0], [0, -1],
                     [1, 1], [-1, 1], [-1, -1], [1, -1]]
            angles = [0., np.pi/2, np.pi, 3*np.pi/2,
                      np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
            s = np.random.choice(range(8))
            x_hat = x_hats[s]
            angle = angles[s]
            x_hat /= np.linalg.norm(x_hat)
            x1, y1 = [xc, yc] - x_hat * interval * num_objects/2
            x2, y2 = [xc, yc] + x_hat * interval * num_objects/2
            if min(x1, x2) < -0.5 or max(x1, x2) > 0.5:
                check_feasible = False
                continue
            if min(y1, y2) < -0.4 or max(y1, y2) > 0.4:
                check_feasible = False
                continue
            check_feasible = True

        xs = np.linspace(x1, x2, num_objects)
        ys = np.linspace(y1, y2, num_objects)
        zs = np.ones(num_objects) * z_default + np.arange(num_objects) * 0.05
        positions = np.concatenate([xs, ys, zs]).reshape(3, num_objects).T
        rotations = [euler2quat([0, 0, angle + rot90*np.pi/2])] * num_objects
    elif scene_type=='circle':
        check_feasible = False
        while not check_feasible:
            radius = np.random.uniform(0.15, 0.3) #0.15
            rot90 = np.random.choice(2)

            x0, y0 = np.array(random_pos_on_table())[:2]
            thetas = np.linspace(0, 1, num_objects+1)[:-1] + np.random.random()
            thetas %= 1
            thetas *= 2 * np.pi
            xs = x0 + radius * np.cos(thetas)
            ys = y0 + radius * np.sin(thetas)
            zs = np.ones(num_objects) * z_default
            if min(xs) < -0.3 or max(xs) > 0.3:
                check_feasible = False
                continue
            if min(ys) < -0.4 or max(ys) > 0.4: 
                check_feasible = False
                continue
            check_feasible = True
        positions = np.concatenate([xs, ys, zs]).reshape(3, num_objects).T
        rotations = [euler2quat([0, 0, th + rot90*np.pi/2]) for th in thetas]
    return positions, rotations

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

def update_visual_objects(object_ids, pkg_path, nv_objects=None, metallic_ids=[], glass_ids=[]):
    # object ids are in pybullet engine
    # pkg_path is for loading the object geometries
    # nv_objects refers to the already entities loaded, otherwise it is going 
    # to load the geometries and create entities. 
    if nv_objects is None:
        nv_objects = { }
    objs = copy.deepcopy(object_ids)
    objs.append(1)
    for object_id in objs:
        is_metallic, is_glass = False, False
        if object_id in metallic_ids:
            is_metallic = True
        elif object_id in glass_ids:
            is_glass = True
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

                if is_metallic:
                    for m in nv_objects[object_name].materials:
                        m.set_base_color((0.5, 0.5, 0.5))
                        m.set_metallic(0.98)
                        m.set_transmission(0)
                        m.set_roughness(0.5)
                elif is_glass:
                    for m in nv_objects[object_name].materials:
                        m.set_base_color((0.6, 0.7, 0.95))
                        m.set_metallic(0.1)
                        m.set_transmission(0.7)
                        m.set_roughness(0.95)
                else:
                    for m in nv_objects[object_name].materials:
                        m.set_base_color((rgbaColor[0] ** 2.2, rgbaColor[1] ** 2.2, rgbaColor[2] ** 2.2))
            # for entities created for primitive shapes
            else:
                if is_metallic:
                    for m in nv_objects[object_name].materials:
                        m.set_base_color((0.3, 0.3, 0.3))
                        m.set_metallic(0.98)
                        m.set_transmission(0)
                        m.set_roughness(0.5)
                elif is_glass:
                    for m in nv_objects[object_name].materials:
                        m.set_base_color(np.array((0.2, 0.3, 0.5))**2)
                        m.set_metallic(0.2)
                        m.set_transmission(0.6)
                        m.set_roughness(0.95)
                else:
                    for m in nv_objects[object_name].materials:
                        m.set_base_color((rgbaColor[0] ** 2.2, rgbaColor[1] ** 2.2, rgbaColor[2] ** 2.2))
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
    #print("check on table & image")
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
    pos_z = np.random.uniform(0.8, 0.9)
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
