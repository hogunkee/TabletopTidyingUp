import json
import os 
import copy
import nvisii as nv
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import pybullet_data
import numpy as np
from matplotlib import pyplot as plt
from gen_sg import generate_sg
from transform_utils import euler2quat, mat2quat, quat2mat
from scene_utils import cal_distance, check_on_table, generate_scene_random, generate_scene_shape, get_init_euler, get_random_pos_orn, move_object, pickable_objects_list, quaternion_multiply, random_pos_on_table
from scene_utils import get_rotation, get_contact_objects, get_velocity
from scene_utils import update_visual_objects 
from scene_utils import remove_visual_objects, clear_scene

obj_name_to_semantic_label = {
    'blue_cup': 'cup',
    'blue_plate': 'plate',
    'blue_tea_box': 'tea_box',
    'book_1': 'book',
    'book_2': 'book',
    'book_3': 'book',
    'book_4': 'book',
    'book_5': 'book',
    'book_6': 'book',
    'book_holder_2': 'book_holder',
    'book_holder_3': 'book_holder',
    'bowl': 'bowl',
    'cleanser': 'cleanser',
    'clear_box': 'basket',
    'clear_box_1': 'basket',
    'conditioner': 'conditioner',
    'cracker_box': 'cracker_box',
    'doraemon_bowl': 'bowl',
    'doraemon_plate': 'tray',
    'extra_large_clamp': 'clamp',
    'flat_screwdriver': 'screwdriver',
    'fork': 'fork',
    'gelatin_box': 'box',
    'glue_1': 'glue',
    'glue_2': 'glue',
    'green_bowl': 'bowl',
    'green_cup': 'cup',
    'grey_plate': 'plate',
    'knife': 'knife',
    'large_clamp': 'clamp',
    'lipton_tea': 'tea_box',
    'medium_clamp': 'clamp',
    'mini_claw_hammer_1': 'hammer',
    'mug': 'mug',
    'orange_cup': 'cup',
    'phillips_screwdriver': 'screwdriver',
    'pink_tea_box': 'tea_box',
    'plastic_apple': 'apple',
    'plastic_banana': 'banana',
    'plastic_lemon': 'lemon',
    'plastic_orange': 'orange',
    'plastic_peach': 'peach',
    'plastic_pear': 'pear',
    'plastic_strawberry': 'strawberry',
    'plate': 'plate',
    'power_drill': 'drill',
    'pudding_box': 'box',
    'round_plate_1': 'plate',
    'round_plate_2': 'plate',
    'round_plate_3': 'plate',
    'round_plate_4': 'plate',
    'scissors': 'scissors',
    'shampoo': 'shampoo',
    'small_clamp': 'clamp',
    'spoon': 'spoon',
    'square_plate_1': 'square_plate',
    'square_plate_2': 'square_plate',
    'square_plate_3': 'square_plate',
    'square_plate_4': 'square_plate',
    'stapler_1': 'stapler',
    'stapler_2': 'stapler',
    'sugar_box': 'box',
    'two_color_hammer': 'hammer',
    'yellow_bowl': 'bowl',
    'yellow_cup': 'cup'
    
}

class TabletopScenes(object):
    def __init__(self, opt):
        self.opt = opt

        # show an interactive window, and use "lazy" updates for faster object creation time 
        nv.initialize(headless=False, lazy_updates=True)

        # Setup bullet physics stuff
        physicsClient = p.connect(p.GUI) # non-graphical version

        # Create a camera
        self.camera = None
        self.set_top_view_camera()
        self.set_grid()

        self.initialize_nvisii_scene()
        self.initialize_pybullet_scene()
        self.init_euler = get_init_euler()
        self.urdf_id_names = self.get_obj_dict(opt.dataset, opt.objectset)

        self.threshold = {'pose': 0.07,
                          'rotation': 0.15,
                          'linear': 0.003,
                          'angular': 0.003}
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.objects_list = {}
        self.spawn_obj_num = 0

    def set_front_top_view_camera(self):
        self.set_camera_pose(eye=(0.5, 0, 1.3), at=(0, 0, 0.3), up=(0, 0, 1))
    
    def set_top_view_camera(self):
        self.set_camera_pose(eye=(0, 0, 1.45), at=(0, 0, 0.3), up=(-1, 0, 0))

    def set_camera_pose(self, eye, at=(0.1, 0, 0), up=(0, 0, 1)):
        if self.camera is None:
            self.camera = nv.entity.create(
                name = "camera",
                transform = nv.transform.create("camera"),
                camera = nv.camera.create_from_fov(
                    name = "camera", field_of_view = 60 * np.pi / 180,
                    aspect = float(self.opt.width)/float(self.opt.height)
                ))

        self.camera.get_transform().look_at(at=at, up=up, eye=eye)
        nv.set_camera_entity(self.camera)
        return


    def initialize_nvisii_scene(self):
        if not self.opt.noise is True: 
            nv.enable_denoiser()

        # Change the dome light intensity
        nv.set_dome_light_intensity(1.0)

        # atmospheric thickness makes the sky go orange, almost like a sunset
        nv.set_dome_light_sky(sun_position=(6,6,6), atmosphere_thickness=1.0, saturation=1.0)

        # add a sun light
        sun = nv.entity.create(
            name = "sun",
            mesh = nv.mesh.create_sphere("sphere"),
            transform = nv.transform.create("sun"),
            light = nv.light.create("sun")
        )
        sun.get_transform().set_position((6,6,6))
        sun.get_light().set_temperature(5780)
        sun.get_light().set_intensity(1000)

        floor = nv.entity.create(
            name="floor",
            mesh = nv.mesh.create_plane("floor"),
            transform = nv.transform.create("floor"),
            material = nv.material.create("floor")
        )
        floor.get_transform().set_position((0,0,0))
        floor.get_transform().set_scale((2, 2, 2)) #(6, 6, 6)
        floor.get_material().set_roughness(0.1)
        floor.get_material().set_base_color((0.8, 0.87, 0.88)) #(0.5,0.5,0.5)

        floor_textures = []
        texture_files = os.listdir("texture")
        texture_files = [f for f in texture_files if f.lower().endswith('.png')]

        for i, tf in enumerate(texture_files):
            tex = nv.texture.create_from_file("tex-%d"%i, os.path.join("texture/", tf))
            floor_tex = nv.texture.create_hsv("floor-%d"%i, tex, hue=0, saturation=.5, value=1.0, mix=1.0)
            floor_textures.append((tex, floor_tex))
        self.floor, self.floor_textures = floor, floor_textures
        return

    def initialize_pybullet_scene(self):
        # reset pybullet #
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setTimeStep(1 / 240)

        # set the plane and table        
        planeID = p.loadURDF("plane.urdf")
        table_id = p.loadURDF("table/table.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071])


        # Set the collision with the floor mesh
        # first lets get the vertices 
        # vertices = self.floor.get_mesh().get_vertices()

        # get the position of the object
        # pos = self.floor.get_transform().get_position()
        # pos = [pos[0],pos[1],pos[2]]
        # scale = self.floor.get_transform().get_scale()
        # scale = [scale[0],scale[1],scale[2]]
        # rot = self.floor.get_transform().get_rotation()
        # rot = [rot[0],rot[1],rot[2],rot[3]]

        # create a collision shape that is a convex hull
        # obj_col_id = p.createCollisionShape(
        #     p.GEOM_MESH,
        #     vertices = vertices,
        #     meshScale = scale,
        # )

        # create a body without mass so it is static
        # p.createMultiBody(
        #     baseCollisionShapeIndex = obj_col_id,
        #     basePosition = pos,
        #     baseOrientation= rot,
        # )    
        return

    def set_grid(self):
        x = np.linspace(-8, -2, 10)
        y = np.linspace(-8, -2, 10)
        xx, yy = np.meshgrid(x, y, sparse=False)
        self.xx = xx.reshape(-1)
        self.yy = yy.reshape(-1)

    def clear(self):

        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        # remove spawned objects before #
        remove_visual_objects(nv.ids)
        #clear_scene()
        self.spawned_objects = None
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.spawn_obj_num = 0
        p.resetSimulation()
        clear_scene()
        self.camera = None
        self.set_top_view_camera()
        self.initialize_nvisii_scene()
        self.initialize_pybullet_scene()
        

    def close(self):
        nv.deinitialize()
        p.disconnect()

    def get_obj_dict(self, dataset, objectset):
        # lets create a bunch of objects 
        pybullet_object_path = self.opt.pybullet_object_path
        pybullet_object_names = sorted([m for m in os.listdir(pybullet_object_path) \
                            if os.path.isdir(os.path.join(pybullet_object_path, m))])
        ycb_object_path = self.opt.ycb_object_path
        ycb_object_names = sorted([m for m in os.listdir(ycb_object_path) \
                            if os.path.isdir(os.path.join(ycb_object_path, m))])
        exclusion_list = ['047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane']
        for eo in exclusion_list:
            if eo in ycb_object_names:
                ycb_object_names.remove(eo)

        pybullet_ids = ['pybullet-%d'%p for p in range(len(pybullet_object_names))]
        ycb_ids = ['ycb-%d'%y for y in range(len(ycb_object_names))]
        # urdf_id_names
        # key: {object_type}-{index} - e.g., 'pybullet-0'
        # value: {object_name} - e.g., 'black_marker'
        if objectset=='pybullet':
            urdf_id_names = dict(zip(
                pybullet_ids, 
                pybullet_object_names
                ))
        elif objectset=='ycb':
            urdf_id_names = dict(zip(
                ycb_ids, 
                ycb_object_names
                ))
        elif objectset=='all':
            urdf_id_names = dict(zip(
                pybullet_ids + ycb_ids,
                pybullet_object_names + ycb_object_names
                ))

        print('-'*60)
        print(len(urdf_id_names), 'objects can be loaded.')
        print('-'*60)
        return urdf_id_names

    # def select_objects(self, nb_objects):
    #     nb_spawn = min(nb_objects, len(self.urdf_id_names))
    #     urdf_selected = np.random.choice(list(self.urdf_id_names.values()), nb_spawn, replace=False)
    #     return urdf_selected

    def spawn_objects(self, spawn_obj_list):
        self.spawned_objects = copy.deepcopy(spawn_obj_list)

        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        pybullet_ids = []
        self.base_rot = {}
        self.base_size = {}
        for obj in spawn_obj_list:
            urdf_id =  urdf_ids[obj_names.index(obj)]
            object_type = urdf_id.split('-')[0]
            #(object_name, object_type) = self.urdf_id_names[urdf_id]
            if object_type=='pybullet':
                urdf_path = os.path.join(self.opt.pybullet_object_path, obj, 'model.urdf')
            else:
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
                if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                    urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'tsdf', 'model.urdf')
                else:
                    urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')

            roll, pitch, yaw = 0, 0, 0
            if urdf_id in self.init_euler:
                roll, pitch, yaw = np.array(self.init_euler[urdf_id]) * np.pi / 2
            rot = get_rotation(roll, pitch, yaw)
            obj_id = p.loadURDF(urdf_path, [self.xx[self.spawn_obj_num], self.yy[self.spawn_obj_num], 0.15], rot, globalScaling=1.) #5.
            for i in range(100):
                p.stepSimulation()
            pos,orn = p.getBasePositionAndOrientation(obj_id)
            self.base_rot[obj_id] = orn
            
            posa, posb = p.getAABB(obj_id)
            # to check object partially on other objects
            self.base_size[obj_id] = (np.abs(posa[0] - posb[0]), np.abs(posa[1] - posb[1]), np.abs(posa[2] - posb[2])) 
            
            pybullet_ids.append(obj_id)
            self.spawn_obj_num += 1
            self.objects_list[int(obj_id)] = obj

        nv.ids = update_visual_objects(pybullet_ids, "")
        self.current_pybullet_ids = copy.deepcopy(pybullet_ids)

    def set_floor(self, texture_id=-1):
        # set floor material #
        roughness = random.uniform(0.1, 0.5)
        self.floor.get_material().clear_base_color_texture()
        self.floor.get_material().set_roughness(roughness)

        if texture_id==-1: # random texture #
            f_cidx = np.random.choice(len(self.floor_textures))
            tex, floor_tex = self.floor_textures[f_cidx]
        else:
            tex, floor_tex = self.floor_textures[texture_id]
        self.floor.get_material().set_base_color_texture(floor_tex)
        self.floor.get_material().set_roughness_texture(tex)

    def arrange_objects(self, scene_id, select=True, random=False):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        # set objects #
        count_scene_trials = 0
        selected_objects = np.random.choice(pybullet_ids, self.opt.inscene_objects, replace=False) if select else pybullet_ids

        while True:
            # generate scene in a 'line' shape #
            if random:  # undefined position and rotation
                pass
                # TODO
                init_positions, init_rotations = generate_scene_random(self.opt.inscene_objects)
                
            else:     # defined position and rotation (rule-base : specific shape or pre-defined : template)
                init_positions, init_rotations = generate_scene_shape(self.opt.scene_type, self.opt.inscene_objects)
                # init_rotations = []
                
                # TODO
                # init_positions, init_rotations = load_scene_template()
            

            #hide objects placed on the table 
            for obj_id in self.pre_selected_objects:
                pos_hidden = [self.xx[pybullet_ids.index(obj_id)], self.yy[pybullet_ids.index(obj_id)], 0.15 ]
                p.resetBasePositionAndOrientation(obj_id, pos_hidden, self.base_rot[obj_id])
            
            # place new objects #
            set_base_rot = True if init_rotations == [] else False               
            for idx, obj_id in enumerate(selected_objects):
                pos_sel = init_positions[idx]
                if set_base_rot:
                    rot = self.base_rot[obj_id]
                    init_rotations.append(rot)
                else: # TODO: set object rotation as template's rotation or random rotation
                    rot = quaternion_multiply(init_rotations[idx], self.base_rot[obj_id]) if random else init_rotations[idx]


                move_object(obj_id, pos_sel, rot)        
                        
            self.pre_selected_objects = []
            init_rotations = np.array(init_rotations)

            is_feasible = self.feasibility_check(selected_objects, init_positions, init_rotations) if not random else self.stable_check(selected_objects)
            data_objects_list = {id: self.objects_list[id] for id in selected_objects}
            is_on_table = check_on_table(data_objects_list)
            is_leaning_obj = self.leaning_check(selected_objects)
            is_feasible = is_feasible and is_on_table and not is_leaning_obj
            
            count_scene_trials += 1
            if is_feasible or count_scene_trials > 5:
                break

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        # if failed to place objects robustly #
        if not is_feasible:
            return False
        
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        self.pickable_objects = []
        self.save_data(scene_id, data_objects_list)
        return True

    # need to change. spherical objects are not considered.
    def feasibility_check(self, selected_objects, init_positions, init_rotations):
        init_feasible = False 
        j = 0
        while j<2000 and not init_feasible:
            p.stepSimulation()
            if j%10==0:
                current_poses = []
                current_rotations = []
                for obj_id in selected_objects:
                    pos, rot = p.getBasePositionAndOrientation(obj_id)
                    current_poses.append(pos)
                    current_rotations.append(rot)

                current_poses = np.array(current_poses)
                current_rotations = np.array(current_rotations)
                pos_diff = np.linalg.norm(current_poses[:, :2] - init_positions[:, :2], axis=1)
                rot_diff = np.linalg.norm(current_rotations - init_rotations, axis=1)
                if (pos_diff > self.threshold['pose']).any() or (rot_diff > self.threshold['rotation']).any():
                    break
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    init_feasible = True
            j += 1
        return init_feasible
    
    def stable_check(self, selected_objects):
        init_stable = False
        j = 0
        while j<1000 and not init_stable:
            p.stepSimulation()
            if j % 10 == 0:
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    init_stable = True
            j+=1
            
        return init_stable

    def leaning_check(self, selected_objects):
        contacts = get_contact_objects()
        is_leaning_obj = False
        leaning_obj_list = []
        for obj in selected_objects:
            contact_table = False
            contact_obj = False
            for s in contacts:
                if obj in s and 1 in s:
                    contact_table = True
                if obj in s and 1 not in s:
                    contact_obj = True
            if contact_table and contact_obj:
                leaning_obj_list.append(obj)
        for obj1 in leaning_obj_list:
            for obj2 in leaning_obj_list:
                if (obj1, obj2) in contacts or (obj2, obj1) in contacts:
                    is_leaning_obj = True
                    break
        return is_leaning_obj
            
                
            
        

    # TODO : change this function to move pickable objects to a random place
    def random_messup_objects(self, scene_id, random_rot = True):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        selected_objects = copy.deepcopy(self.pre_selected_objects)
        data_objects_list = {id: self.objects_list[id] for id in selected_objects}

        #select pickable object
        move_obj_id = np.random.choice(self.pickable_objects)
        
        # save current poses & rots #
        pos_saved, rot_saved = {}, {}
        for obj_id in selected_objects:
            pos, rot = p.getBasePositionAndOrientation(obj_id)
            pos_saved[obj_id] = pos
            rot_saved[obj_id] = rot

        # set poses & rots #
        
        place_feasible = False
        count_scene_repose = 0
        while not place_feasible:
            # get the pose of the objects
            pos, rot = p.getBasePositionAndOrientation(move_obj_id)
            collisions_before = get_contact_objects()

            if random_rot:
                pos_new, rot = get_random_pos_orn(rot)
            else :
                pos = random_pos_on_table()
                rot = self.base_rot[move_obj_id]
            
            p.resetBasePositionAndOrientation(move_obj_id, pos_new, rot)
            collisions_after = set()
            for _ in range(200):
                p.stepSimulation()
                collisions_after = collisions_after.union(get_contact_objects())

            collisions_new = collisions_after - collisions_before
            if len(collisions_new) > 0:
                place_feasible = False

                # reset non-target objects
                obj_to_reset = set()
                for collision in collisions_new:
                    obj1, obj2 = collision
                    obj_to_reset.add(obj1)
                    obj_to_reset.add(obj2)
                obj_to_reset = obj_to_reset - set([move_obj_id]) - set([1]) # table id
                for reset_obj_id in obj_to_reset:
                    p.resetBasePositionAndOrientation(reset_obj_id, pos_saved[reset_obj_id], rot_saved[reset_obj_id])
            else:
                place_feasible = True
                pos, rot = p.getBasePositionAndOrientation(move_obj_id)
                pos_saved[move_obj_id] = pos
                rot_saved[move_obj_id] = rot
            count_scene_repose += 1
            is_on_table = check_on_table(data_objects_list)
            is_leaning_obj = self.leaning_check(selected_objects)
            place_feasible = place_feasible and is_on_table and not is_leaning_obj

            if count_scene_repose > 10:
                break

        if not place_feasible:
            return False

        for j in range(2000):
            p.stepSimulation()
            if j%10==0:
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    break
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        
        self.save_data(scene_id, data_objects_list)
        return True

    def render_and_save_scene(self,out_folder):
        nv.render_to_file(
            width=int(self.opt.width), height=int(self.opt.height), 
            samples_per_pixel=int(self.opt.spp),
            file_path=f"{out_folder}/rgb.png"
        )
        d = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=5, bounce=0, options='depth',
        )
        depth = np.array(d).reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        depth = np.flip(depth, axis = 0)
        depth[np.isinf(depth)] = 3
        depth[depth < 0] = 3
        np.save(f"{out_folder}/depth.npy", depth)
        
        entity_id = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=5, bounce=0, options='entity_id',
        )
        entity = np.array(entity_id)
        entity = entity.reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        entity = entity - 1
        entity = np.flip(entity, axis = 0)
        entity[np.isinf(entity)] = -1
        entity[entity>self.opt.nb_objects + 50] = -1
        np.save(f"{out_folder}/seg.npy", entity)
        return

    def save_data(self, scene_idx, objects_list):
        """Save data to a file."""
        obj_info = {}
        out_folder = f"{self.opt.out_folder}/{self.opt.save_scene_name}/template_{str(scene_idx['template']).zfill(5)}/traj_{str(scene_idx['trajectory']).zfill(5)}/{str(scene_idx['frame']).zfill(3)}"
        if os.path.isdir(out_folder):
            print(f'folder {out_folder}/ exists')
        else:
            os.makedirs(out_folder)
            print(f'created folder {out_folder}/')
        self.render_and_save_scene(out_folder)
        
        obj_semantic_label = {}
        obj_aabb = {}
        obj_adjacency = {}
        obj_state = {}
        n_obj = {}
        new_objects_list = {}
        for id,obj_name in objects_list.items():
            id = int(id)
            new_objects_list[id] = obj_name
            n = n_obj.get(obj_name_to_semantic_label[obj_name], 0)
            n_obj[obj_name_to_semantic_label[obj_name]] = n + 1
            obj_semantic_label[id] = obj_name_to_semantic_label[obj_name] + '_' + str(n)
            obj_aabb[id] = p.getAABB(id)
            obj_state[id] = p.getBasePositionAndOrientation(id)
            overlapping_objs = p.getOverlappingObjects(obj_aabb[id][0], obj_aabb[id][1])
            obj_adjacency[id] = []
            for obj in overlapping_objs:
                obj_adjacency[id].append(obj[0])
        
        obj_info['objects'] = new_objects_list
        obj_info['semantic_label'] = obj_semantic_label
        obj_info['obj_aabb'] = obj_aabb
        obj_info['distances'] = cal_distance(new_objects_list)
        obj_info['state'] = obj_state
        sg = generate_sg(obj_info)
        obj_info['pickable_objects'] = pickable_objects_list(new_objects_list, sg)
        obj_info['scene_graph'] = sg
        self.pickable_objects = obj_info['pickable_objects']
        with open(out_folder+"/obj_info.json", "w") as f:
            json.dump(obj_info, f)
        
        return





if __name__=='__main__':
    opt = lambda : None
    opt.nb_objects = 12 #20
    opt.inscene_objects = 4 #5
    opt.scene_type = 'line' # 'random' or 'line'
    opt.save_scene_name = 'test'
    opt.spp = 32 #64 
    opt.width = 480
    opt.height = 360
    opt.noise = False
    opt.nb_scenes_per_set = 5
    opt.nb_frames = 5
    opt.out_folder = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/dataset'
    opt.nb_randomset = 5
    opt.num_traj = 20
    opt.dataset = 'train' #'train' or 'test'
    opt.objectset = 'pybullet' #'pybullet'/'ycb'/'all'
    opt.pybullet_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
    opt.ycb_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/YCB_dataset'

    if os.path.isdir(opt.out_folder):
        print(f'folder {opt.out_folder}/ exists')
    else:
        os.makedirs(opt.out_folder)
        print(f'created folder {opt.out_folder}/')
        
    #pre defined objects to spawn. 
    #if you want, you can get the list of objects using select_objects. (need to change urdf_ids to urdf_names)
    spawn_objects_list = ['bowl', 'spoon', 'knife', 'blue_cup', 'book_1', 'book_2', 'book_3', 'cleanser', 'conditioner', 'doraemon_bowl', 'fork', 'green_bowl']
    
    ts = TabletopScenes(opt)
    for n_set in range(opt.nb_randomset): # line, circle -> random set, template -> set of objects in the template.
        ts.spawn_objects(spawn_objects_list) # add random select or template load
        # urdf_selected = ts.select_objects(opt.nb_objects)
        for i in range(opt.num_traj):
            # TODO : change this part##################
            # exist_trajs = ~~
            # num_exist_trajs = ~~
            traj_id = i # + num_exist_trajs
            #############################
            
            scene_id = {'template': n_set,'trajectory': traj_id, 'frame': 0}
            # 1. Spawn objects on the table (line shape or load from template) #
            success_placement = False
            while not success_placement:
                ts.set_floor(texture_id=-1)
                print(f'rendering scene {str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
                success_placement = ts.arrange_objects(scene_id, random=True)
                if not success_placement:
                    continue
            scene_id['frame'] += 1
            
            # 2. Move each object to a random place #
            while scene_id['frame'] < int(opt.nb_frames): #
                print(f'rendering scene {str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
                success_placement = ts.random_messup_objects(scene_id)
                if not success_placement:
                    continue
                scene_id['frame'] += 1
                
        ts.clear()
    ts.close()

    
