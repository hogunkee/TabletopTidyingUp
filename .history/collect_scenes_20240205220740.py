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
from scene_utils import cal_distance, check_on_table, generate_scene_random, generate_scene_shape, get_init_euler, get_random_pos_from_grid, get_random_pos_orn, move_object, pickable_objects_list, quaternion_multiply, random_pos_on_table
from scene_utils import get_rotation, get_contact_objects, get_velocity
from scene_utils import update_visual_objects 
from scene_utils import remove_visual_objects, clear_scene
from scene_utils import get_object_categories
from collect_template_list import cat_to_name_test, cat_to_name_train
from itertools import product

class TabletopScenes(object):
    def __init__(self, opt, data_collect = False):
        self.opt = opt

        # show an interactive window, and use "lazy" updates for faster object creation time 
        nv.initialize(headless=True, lazy_updates=True) # headless=False

        # Setup bullet physics stuff
        physicsClient = p.connect(p.GUI) # non-graphical version

        # Create a camera
        self.camera_top = None
        self.camera_front_top = None
        self.set_top_view_camera()
        self.set_front_top_view_camera()
        self.set_grid()

        self.initialize_nvisii_scene()
        self.initialize_pybullet_scene()
        self.init_euler = get_init_euler()
        self.urdf_id_names = self.get_obj_dict(opt.dataset, opt.objectset)

        self.threshold = {'pose': 0.07,
                          'rotation': 0.25, #0.15,
                          'linear': 0.003,
                          'angular': 0.03} # angular : 0.003 (too tight to random collect)
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.objects_list = {}
        self.spawn_obj_num = 0
        if data_collect:
            self.data_collect_init()

    def data_collect_init(self):
        # object_cat_to_name = get_object_categories()
        
        if self.opt.object_split=='unseen':
            self.cat_to_name = cat_to_name_test
            for cat in self.cat_to_name.keys():
                if not self.cat_to_name[cat]:
                    self.cat_to_name[cat].append(cat_to_name_train[cat][0])
        else:
            self.cat_to_name = cat_to_name_train
            
        self.obj_name_to_semantic_label = {}
        self.object_name_list = []
        for cat, names in self.cat_to_name.items():
            self.object_name_list += names
            for name in names:
                n = name.split('/')[-1]
                self.obj_name_to_semantic_label[n] = cat



    def set_front_top_view_camera(self):
        self.camera_front_top = self.set_camera_pose(eye=(0.5, 0, 1.3), at=(0, 0, 0.3), up=(0, 0, 1), view = 'front_top')
    
    def set_top_view_camera(self):
        self.camera_top = self.set_camera_pose(eye=(0, 0, 1.45), at=(0, 0, 0.3), up=(-1, 0, 0))

    def set_camera_pose(self, eye, at=(0.1, 0, 0), up=(0, 0, 1), view='top'):
        camera = nv.entity.create(
            name = "camera_" + view,
            transform = nv.transform.create("camera_" + view),
            camera = nv.camera.create_from_fov(
                name = "camera_" + view, field_of_view = 60 * np.pi / 180,
                aspect = float(self.opt.width)/float(self.opt.height)
            ))

        camera.get_transform().look_at(at=at, up=up, eye=eye)
        # nv.set_camera_entity(camera)
        return camera


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
        table_id = p.loadURDF("table/table2.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071])
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
        self.camera_top = None
        self.camera_front_top = None
        self.set_top_view_camera()
        self.set_front_top_view_camera()
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
        if '__pycache__' in pybullet_object_names:
            pybullet_object_names.remove('__pycache__')
        ycb_object_path = self.opt.ycb_object_path
        ycb_object_names = sorted([m for m in os.listdir(ycb_object_path) \
                            if os.path.isdir(os.path.join(ycb_object_path, m))])
        # exclusion_list = ['047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane', '033_spatula', '39_key']
        exclusion_list = ['039_key', '046_plastic_bolt', '047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane']
        for eo in exclusion_list:
            if eo in ycb_object_names:
                ycb_object_names.remove(eo)
        # ig_object_path = self.opt.ig_object_path
        # ig_object_names = []
        # for m1 in os.listdir(ig_object_path):
        #     for m2 in os.listdir(os.path.join(ig_object_path, m1)):
        #         ig_object_names.append(os.path.join(m1, m2))
        housecat_object_path = self.opt.housecat_object_path
        housecat_categories = [h for h in os.listdir(housecat_object_path) if os.path.isdir(os.path.join(housecat_object_path, h)) and not h in ['bg', 'collision']]
        housecat_object_names = []
        for m1 in housecat_categories:
            obj_model_list = [h for h in os.listdir(os.path.join(housecat_object_path, m1)) if h.endswith('.urdf')]
            for m2 in obj_model_list:
                housecat_object_names.append(m2.split('.urdf')[0])
        housecat_object_names = sorted(housecat_object_names)
        
        pybullet_ids = ['pybullet-%d'%p for p in range(len(pybullet_object_names))]
        ycb_ids = ['ycb-%d'%y for y in range(len(ycb_object_names))]
        # ig_ids = ['ig-%d'%i for i in range(len(ig_object_names))]
        housecat_ids = ['housecat-%d'%h for h in range(len(housecat_object_names))]
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
        # elif objectset=='ig':
        #     urdf_id_names = dict(zip(
        #         ig_ids,
        #         ig_object_names
        #         ))
        elif objectset=='housecat':
            urdf_id_names = dict(zip(
                housecat_ids,
                housecat_object_names
                ))
        elif objectset=='all':
            urdf_id_names = dict(zip(
                pybullet_ids + ycb_ids + housecat_ids,
                pybullet_object_names + ycb_object_names + housecat_object_names
                ))

        print('-'*40)
        print(len(urdf_id_names), 'objects can be loaded.')
        print('-'*40)
        return urdf_id_names

    def spawn_objects(self, spawn_obj_list):  # spawn_obj_list : [(obj_name, size), ...]
        self.spawned_objects = copy.deepcopy(spawn_obj_list)
        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        metallic_objects = ['cutlery-fork_1', 'cutlery-fork_1_new', 'cutlery-fork_2_new', 'cutlery-fork_3_new', 'cutlery-knife_1', 'cutlery-knife_1_new', 'cutlery-knife_2', 'cutlery-knife_2_new', 'cutlery-knife_3_new', 'cutlery-spoon_1', 'cutlery-spoon_1_new', 'cutlery-spoon_2', 'cutlery-spoon_2_new', 'cutlery-spoon_3_new', 'cutlery-spoon_4_new', 'cutlery-spoon_5_new']
        glass_objects = ['glass-new_11', 'glass-new_7', 'glass-small', 'glass-new_1', 'glass-new_9', 'glass-new_8', 'glass-new_13', 'glass-new_10', 'glass-new_3', 'glass-small_4', 'glass-new_4', 'glass-new_6', 'glass-small_3', 'glass-new_12', 'glass-cocktail', 'glass-new_5', 'glass-new_2']
        self.metallic_ids = []
        self.glass_ids = []
        pybullet_ids = []
        self.base_rot = {}
        self.base_size = {}
        for obj_, size in spawn_obj_list: ### TODO: 사이즈별로 (large, small) 에 따라서 크기 다르게 spawn해야함.
            object_type = obj_.split('/')[0]
            obj = obj_.split('/')[-1]
            urdf_id = urdf_ids[obj_names.index(obj)]
            # object_type = urdf_id.split('-')[0]
            if object_type=='pybullet':
                urdf_path = os.path.join(self.opt.pybullet_object_path, obj, 'model.urdf')
            elif object_type=='ycb':
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
                if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                    urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'tsdf', 'model.urdf')
                else:
                    urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
            # elif object_type=='ig':
            #     urdf_path = os.path.join(self.opt.ig_object_path, obj, obj.split('/')[-1]+'.urdf')
            elif object_type=='housecat':
                urdf_path = os.path.join(self.opt.housecat_object_path, obj.split('-')[0], obj+'.urdf')

            roll, pitch, yaw = 0, 0, 0
            if urdf_id in self.init_euler:
                roll, pitch, yaw, scale = np.array(self.init_euler[urdf_id])
                if size == 'large': scale = scale * 1.2
                elif size == 'small': scale = scale * 0.7
                roll, pitch, yaw = roll * np.pi / 2, pitch * np.pi / 2, yaw * np.pi / 2
            rot = get_rotation(roll, pitch, yaw)
            obj_id = p.loadURDF(urdf_path, [self.xx[self.spawn_obj_num], self.yy[self.spawn_obj_num], 0.15], rot, globalScaling=scale) #5.
            for i in range(100):
                p.stepSimulation()
            pos,orn = p.getBasePositionAndOrientation(obj_id)
            self.base_rot[obj_id] = orn
            
            posa, posb = p.getAABB(obj_id)
            # to check object partially on other objects
            self.base_size[obj_id] = (np.abs(posa[0] - posb[0]), np.abs(posa[1] - posb[1]), np.abs(posa[2] - posb[2])) 
            
            pybullet_ids.append(obj_id)
            self.spawn_obj_num += 1
            self.objects_list[int(obj_id)] = (obj, size)

            if obj in metallic_objects:
                self.metallic_ids.append(obj_id)
            if obj in glass_objects:
                self.glass_ids.append(obj_id)

        nv.ids = update_visual_objects(pybullet_ids, "", metallic_ids=self.metallic_ids, glass_ids=self.glass_ids)
        self.current_pybullet_ids = copy.deepcopy(pybullet_ids)

    def respawn_object(self, obj_, scale=1.):
        spawn_idx = self.spawned_objects.index(obj_)
        pybullet_ids = self.current_pybullet_ids
        obj_id_to_remove = pybullet_ids[spawn_idx]

        # remove the existing object
        self.base_rot.pop(obj_id_to_remove)
        self.base_size.pop(obj_id_to_remove)
        self.objects_list.pop(int(obj_id_to_remove))
        p.removeBody(obj_id_to_remove)

        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        object_type = obj_[0].split('/')[0]
        obj = obj_[0].split('/')[-1]
        
        urdf_id = urdf_ids[obj_names.index(obj)]
        if object_type=='pybullet':
            urdf_path = os.path.join(self.opt.pybullet_object_path, obj, 'model.urdf')
        elif object_type=='ycb':
            urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
            if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'tsdf', 'model.urdf')
            else:
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
        # elif object_type=='ig':
        #     urdf_path = os.path.join(self.opt.ig_object_path, obj, obj.split('/')[-1]+'.urdf')
        elif object_type=='housecat':
            urdf_path = os.path.join(self.opt.housecat_object_path, obj.split('-')[0], obj+'.urdf')

        roll, pitch, yaw = 0, 0, 0
        if urdf_id in self.init_euler:
            roll, pitch, yaw, _ = np.array(self.init_euler[urdf_id]) * np.pi / 2
        rot = get_rotation(roll, pitch, yaw)
        obj_id = p.loadURDF(urdf_path, [self.xx[self.spawn_obj_num], self.yy[self.spawn_obj_num], 0.15], rot, globalScaling=scale)
        for i in range(100):
            p.stepSimulation()
        pos,orn = p.getBasePositionAndOrientation(obj_id)
        self.base_rot[obj_id] = orn
        
        posa, posb = p.getAABB(obj_id)
        # to check object partially on other objects
        self.base_size[obj_id] = (np.abs(posa[0] - posb[0]), np.abs(posa[1] - posb[1]), np.abs(posa[2] - posb[2])) 
        
        pybullet_ids[spawn_idx] = obj_id
        self.objects_list[int(obj_id)] = (obj, obj_[1])

        nv.ids = update_visual_objects(pybullet_ids, "", metallic_ids=self.metallic_ids, glass_ids=self.glass_ids)
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

    def arrange_objects(self, scene_id, select=True, random=False ):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        # set objects #
        count_scene_trials = 0
        selected_objects = np.random.choice(pybullet_ids, self.opt.inscene_objects, replace=False) if select else pybullet_ids

        while True:
            #hide objects placed on the table 
            for obj_id in self.pre_selected_objects:
                pos_hidden = [self.xx[pybullet_ids.index(obj_id)], self.yy[pybullet_ids.index(obj_id)], 0.15 ]
                p.resetBasePositionAndOrientation(obj_id, pos_hidden, self.base_rot[obj_id])

            # generate scene in a 'line' shape #
            if random:  # undefined position and rotation
                init_positions, init_rotations = generate_scene_random(self.opt.inscene_objects)
            else:     # defined position and rotation (rule-based : specific shape)
                init_positions, init_rotations = generate_scene_shape(self.opt.scene_type, self.opt.inscene_objects)
                # init_rotations = []                
            is_feasible, data_objects_list = self.load_obj_without_template(selected_objects, init_positions, init_rotations)
            
            count_scene_trials += 1
            if is_feasible or count_scene_trials > 5:
                break

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        # if failed to place objects robustly #
        if not is_feasible:
            return False
        
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids, metallic_ids=self.metallic_ids, glass_ids=self.glass_ids)
        self.pickable_objects = []
        self.save_data(scene_id, data_objects_list)
        return True

    def load_obj_without_template(self, selected_objects, init_positions, init_rotations):
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

            # calculate distance btw objects and repose objects
            if self.opt.scene_type=='line':
                goal_dist = 0.02
                if idx==0:
                    continue
                pre_obj_id = selected_objects[idx-1]
                #pre_obj_id = copy.deepcopy(obj_id)
                objects_pair = {
                        pre_obj_id: self.objects_list[pre_obj_id][0],
                        obj_id: self.objects_list[obj_id][0]
                        }
                distance = list(cal_distance(objects_pair).values())[0]
                if distance < 0.0001:
                    continue
                # delta = dist(o1, o2) - goal_dist
                # x_hat := unit vector from o1 to o2
                # p2_new = p2 - delta * x_hat
                pre_pos_sel = init_positions[idx-1]
                x_hat = pos_sel - pre_pos_sel
                x_hat /= np.linalg.norm(x_hat)
                pos_new = pos_sel - (distance - goal_dist) * x_hat
                move_object(obj_id, pos_new, rot)
                init_positions[idx] = pos_new
                #print(cal_distance(objects_pair))

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        init_rotations = np.array(init_rotations)

        is_feasible = self.feasibility_check(selected_objects, init_positions, init_rotations) if not random else self.stable_check(selected_objects)
        data_objects_list = {id: self.objects_list[id] for id in selected_objects}
        is_on_table = check_on_table(data_objects_list)
        is_leaning_obj = self.leaning_check(selected_objects)
        is_feasible = is_feasible and is_on_table and not is_leaning_obj

        return is_feasible, data_objects_list

    def load_template(self, scene_id, templates, objects=None): 
        self.pre_selected_objects = []
        template_id_to_sim_id = {}
        data_objects_list = {}
        template_id_to_obj = {}
        spawn_list = []
        if objects is None:
            for action in templates['action_order']:
                action_type, obj_id, pos, rot = action
                if action_type == 0: # spawn
                    obj_cat, obj_size = templates['objects'][str(obj_id)]
                    obj_name = np.random.choice(self.cat_to_name[obj_cat])  
                    spawn_list.append((obj_name, obj_size))
                    obj_name = obj_name.split('/')[-1]
                    template_id_to_obj[obj_id] = (obj_name, obj_size)
        else:
            for i in range(len(objects)):
                obj_type, obj_name, obj_size, obj_id = objects[i]
                spawn_list.append((obj_type + '/' + obj_name, obj_size))
                template_id_to_obj[obj_id] = (obj_name, obj_size)
        
        self.spawn_objects(spawn_list)
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        for action in templates['action_order']:
            action_type, obj_id, pos, rot = action
            if action_type == 0: # spawn
                obj = template_id_to_obj[obj_id]
                key_list = list(self.objects_list.keys())
                value_list = list(self.objects_list.values())
                sim_obj_id = key_list[value_list.index(obj)]
                template_id_to_sim_id[obj_id] = sim_obj_id
                data_objects_list[sim_obj_id] = self.objects_list[sim_obj_id]
                spawn_rot = quaternion_multiply(rot, self.base_rot[sim_obj_id])
                p.resetBasePositionAndOrientation(sim_obj_id, pos, spawn_rot)
                
            elif action_type == 1:
                sim_obj_id = template_id_to_sim_id[obj_id]
                orig_pos, orig_rot = p.getBasePositionAndOrientation(sim_obj_id)
                rot = quaternion_multiply(rot, orig_rot)
                p.resetBasePositionAndOrientation(sim_obj_id, pos, rot)                
                
            elif action_type == 2:
                for _ in range(100):
                    p.stepSimulation()
        self.pre_selected_objects = list(template_id_to_sim_id.values())
        # TODO : check feasibility if needed

        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids, metallic_ids=self.metallic_ids, glass_ids=self.glass_ids)

        self.save_data(scene_id, data_objects_list)

    def get_obj_combinations(self, template):
        add_obj_list = []
        for action in template['action_order']:
            action_type, obj_id, pos, rot = action
            if action_type == 0:
                obj_cat, obj_size = template['objects'][str(obj_id)]
                obj_list = []
                for obj_name in self.cat_to_name[obj_cat]:
                    obj_type, obj_name_ = obj_name.split('/')
                    obj_list.append((obj_type, obj_name_, obj_size, obj_id))
                    
                add_obj_list.append(obj_list)
        return list(product(*add_obj_list))
    
    def get_augmented_templates(self, template, num_augmentations=5):
        final_poses = {}
        objects = []
        for action in template['action_order']:
            action_type, obj_id, pos, rot = action
            if pos is not None:
                if obj_id not in objects:
                    objects.append(obj_id)
                final_poses[obj_id] = pos
        new_templates = []
        new_template = copy.deepcopy(template)
        new_templates.append(new_template)
        mean_pos_x = np.mean([final_poses[obj_id][0] for obj_id in objects])
        mean_pos_y = np.mean([final_poses[obj_id][1] for obj_id in objects])
        
        for i in range(num_augmentations-1):
            new_template = copy.deepcopy(template)
            while True:
                is_good = True
                random_pos_diff = np.random.uniform(-0.15, 0.15,  2)
                random_scaling = np.random.uniform(0.7, 1.1)
                pos_diff = {}
                for obj_id in objects:
                    pos_x = (final_poses[obj_id][0] - mean_pos_x) * random_scaling + mean_pos_x + random_pos_diff[0]
                    pos_y = (final_poses[obj_id][1] - mean_pos_y) * random_scaling + mean_pos_y + random_pos_diff[1]
                    pos_diff[obj_id] = (pos_x - final_poses[obj_id][0], pos_y - final_poses[obj_id][1])
                    if(-0.4 < pos_x < 0.4 and -0.55 < pos_y < 0.55):
                        continue
                    else:
                        is_good = False
                if is_good:
                    break
            for j,action in enumerate(new_template['action_order']):
                action_type, obj_id, pos, rot = action
                if pos is not None:
                    new_pos = [pos[0] + pos_diff[obj_id][0], pos[1] + pos_diff[obj_id][1] , pos[2]]
                    new_template['action_order'][j] = (action_type, obj_id, new_pos, rot)
            new_templates.append(new_template)
                        
        return new_templates
    
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

            if random_rot:
                pos_new, rot = get_random_pos_orn(rot)
            else :
                pos = random_pos_on_table()
                rot = self.base_rot[move_obj_id]
            if self.opt.mess_grid:
                pos = get_random_pos_from_grid()
            p.resetBasePositionAndOrientation(move_obj_id, pos_new, rot)

            for _ in range(200):
                p.stepSimulation()


            feasible = True
            for obj_id in selected_objects:
                # reset non-target objects
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                pos_diff = np.linalg.norm(np.array(pos[:2]) - np.array(pos_saved[obj_id][:2]))
                r_inv = (-rot_saved[obj_id][0], -rot_saved[obj_id][1], -rot_saved[obj_id][2], rot_saved[obj_id][3])
                rot_inv = p.getEulerFromQuaternion(quaternion_multiply(rot, r_inv))
                rot_diff = np.linalg.norm(np.array(rot) - np.array(rot_saved[obj_id]))
                rot_diff2 = np.linalg.norm(np.array(rot_inv[:2]))

                if obj_id == move_obj_id:
                    if rot_diff2 > 0.2:
                        feasible = False
                        p.resetBasePositionAndOrientation(obj_id, pos_saved[obj_id], rot_saved[obj_id])
                    continue

                

                if (pos_diff > self.threshold['pose']).any() or (rot_diff > self.threshold['rotation']).any() or rot_diff2 > 0.2:
                    feasible = False
                    p.resetBasePositionAndOrientation(obj_id, pos_saved[obj_id], rot_saved[obj_id])
            
            if feasible:
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
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids, metallic_ids=self.metallic_ids, glass_ids=self.glass_ids)
        
        self.save_data(scene_id, data_objects_list)
        return True

    def render_and_save_scene(self,out_folder, camera):
        if camera == 'top':
            nv.set_camera_entity(self.camera_top)
        elif camera == 'front_top':
            nv.set_camera_entity(self.camera_front_top)
        nv.render_to_file(
            width=int(self.opt.width), height=int(self.opt.height), 
            samples_per_pixel=int(self.opt.spp),
            file_path=f"{out_folder}/rgb_{camera}.png"
        )
        d = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=5, bounce=0, options='depth',
        )
        depth = np.array(d).reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        depth = np.flip(depth, axis = 0)
        depth[np.isinf(depth)] = 3
        depth[depth < 0] = 3
        depth = depth.astype(np.float16)
        np.save(f"{out_folder}/depth_{camera}.npy", depth)
        
        entity_id = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=1, bounce=0, options='entity_id',
        )
        entity = np.array(entity_id)
        entity = entity.reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        entity = entity - 2
        entity = np.flip(entity, axis = 0)
        entity[np.isinf(entity)] = -1
        entity[entity>self.opt.nb_objects + 50] = -1
        entity = entity.astype(np.int8)
        np.save(f"{out_folder}/seg_{camera}.npy", entity)
        return

    def save_data(self, scene_idx, objects_list):
        """Save data to a file."""
        obj_info = {}
        out_folder = f"{self.opt.out_folder}/{self.opt.dataset}/{scene_idx['scene']}/template_{str(scene_idx['template_id']).zfill(5)}/traj_{str(scene_idx['trajectory']).zfill(5)}/{str(scene_idx['frame']).zfill(3)}"
        # out_folder = f"{self.opt.out_folder}/{self.opt.save_scene_name}/template_{str(scene_idx['template']).zfill(5)}/traj_{str(scene_idx['trajectory']).zfill(5)}/{str(scene_idx['frame']).zfill(3)}"
        if os.path.isdir(out_folder):
            print(f'folder {out_folder}/ exists')
        else:
            os.makedirs(out_folder)
            print(f'created folder {out_folder}/')
        self.render_and_save_scene(out_folder,'top')
        self.render_and_save_scene(out_folder,'front_top')
        
        obj_semantic_label = {}
        obj_aabb = {}
        obj_adjacency = {}
        obj_state = {}
        n_obj = {}
        sizes = {}
        new_objects_list = {}
        for id, (obj_name, size) in objects_list.items():
            id = int(id)
            new_objects_list[id] = obj_name
            n = n_obj.get(self.obj_name_to_semantic_label[obj_name], 0)
            n_obj[self.obj_name_to_semantic_label[obj_name]] = n + 1
            obj_semantic_label[id] = self.obj_name_to_semantic_label[obj_name] + '_' + str(n)
            sizes[id] = size
            obj_aabb[id] = p.getAABB(id)
            obj_state[id] = p.getBasePositionAndOrientation(id)
            overlapping_objs = p.getOverlappingObjects(obj_aabb[id][0], obj_aabb[id][1])
            obj_adjacency[id] = []
            for obj in overlapping_objs:
                obj_adjacency[id].append(obj[0])
        
        obj_info['objects'] = new_objects_list
        obj_info['semantic_label'] = obj_semantic_label
        obj_info['obj_aabb'] = obj_aabb
        obj_info['sizes'] = sizes
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
    opt.nb_objects = 15 #20
    opt.inscene_objects = 5 #7
    opt.scene_type = 'random' # 'random' or 'line'
    opt.spp = 32 #64 
    opt.width = 480
    opt.height = 360
    opt.noise = False
    opt.mess_grid = True
    opt.nb_frames = 5 #7
    # opt.out_folder = '/ssd/disk/ur5_tidying_data/template-test/'
    # opt.out_folder = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/dataset'
    opt.out_folder = '/home/brain2/workspace/TabletopTidyingUp/dataset'
    opt.nb_randomset = 50
    opt.num_traj = 40
    opt.num_combinations = 20
    opt.dataset = 'train' #'train' or 'test'
    opt.object_split = 'seen' # 'unseen' or 'seen'
    opt.scene_split = 'seen' # 'unseen' or 'seen'
    opt.objectset = 'all' # 'pybullet' #'pybullet'/'ycb'/ 'housecat'/ 'all'
    # opt.pybullet_object_path = '/ssd/disk/pybullet-URDF-models/urdf_models/models'
    # opt.ycb_object_path = '/ssd/disk/YCB_dataset'
    # opt.housecat_object_path = '/ssd/disk/housecat6d/obj_models_small_size_final'
    opt.ig_object_path = '/ssd/disk/ig_dataset/objects'
    opt.pybullet_object_path = '/home/b/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
    opt.ycb_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/YCB_dataset'
    opt.housecat_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/housecat6d/obj_models_small_size_final'

    if os.path.isdir(opt.out_folder):
        print(f'folder {opt.out_folder}/ exists')
    else:
        os.makedirs(opt.out_folder)
        print(f'created folder {opt.out_folder}/')
      
    ### use template ###
    # if 'unseen' in [opt.scene_split, opt.object_split]:
    #     opt.dataset = f'test-{opt.object_type}_obj-{opt.template_type}_template'
    # else : 
    #     opt.dataset = 'train'

    # template_folder = './templates'
    # template_files = os.listdir(template_folder)
    # template_files = [f for f in template_files if f.lower().endswith('.json')]
    # collect_scenes = ['D11'] #... train/test 나눠서 수집. test에 들어가는거 : unseen template, unseen obj + seen template. 이거 두개도 나눠서 수집해야할듯,
    # ts = TabletopScenes(opt, data_collect=True)
    # for template_file in template_files:
    #     if template_file.split('_')[0] in collect_scenes:
    #         traj_id = 0
    #         scene = template_file.split('_')[0]
    #         template_id = template_file.split('_')[-1].split('.')[0]
    #         with open(os.path.join(template_folder, template_file), 'r') as f:
    #             templates = json.load(f)
                
    #         for i in range(opt.num_combinations):   # 20 << total combinations. (random select)
    #             augmented_templates = ts.get_augmented_templates(templates)
    #             for augmented_template in augmented_templates:
    #                 scene_id = {'scene': scene, 'template_id': template_id,'trajectory': traj_id, 'frame': 0}

    #                 # 1. Load template #
    #                 ts.set_floor(texture_id=-1)
    #                 print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
    #                 success_placement = ts.load_template(scene_id, augmented_template)
    #                 scene_id['frame'] += 1
            
    #                 # 2. Move each object to a random place #
    #                 while scene_id['frame'] < int(opt.nb_frames): #
    #                     print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
    #                     success_placement = ts.random_messup_objects(scene_id)
    #                     if not success_placement:
    #                         continue
    #                     scene_id['frame'] += 1
    #                 traj_id += 1
                
    #                 ts.clear()
    # ts.close()

    scenes = ['random_4', 'random_5', 'random_6', 'random_7'] #train 
    # scenes = ['random_5', 'random_6', 'random_7', 'random_8'] #test
    ts = TabletopScenes(opt, data_collect=True)
    spawn_objects_list = ts.object_name_list #['stapler_2', 'two_color_hammer', 'scissors', 'extra_large_clamp', 'phillips_screwdriver', 'stapler_1', 'conditioner', 'book_1', 'book_2', 'book_3', 'book_4', 'book_5', 'book_6', 'power_drill', 'plastic_pear', 'cracker_box', 'blue_plate', 'blue_cup', 'cleanser', 'bowl', 'plastic_lemon', 'mug', 'square_plate_4', 'sugar_box', 'plastic_strawberry', 'medium_clamp', 'plastic_peach', 'knife', 'square_plate_2', 'fork', 'plate', 'green_cup', 'green_bowl', 'orange_cup', 'large_clamp', 'spoon', 'pink_tea_box', 'pudding_box', 'plastic_orange', 'plastic_apple', 'doraemon_plate', 'lipton_tea', 'yellow_bowl', 'grey_plate', 'gelatin_box', 'blue_tea_box', 'flat_screwdriver', 'mini_claw_hammer_1', 'shampoo', 'glue_1', 'glue_2', 'small_clamp', 'square_plate_3', 'doraemon_bowl', 'square_plate_1', 'round_plate_1', 'round_plate_3', 'round_plate_2', 'round_plate_4', 'plastic_banana', 'yellow_cup']
    for scene in scenes:
        for n_set in range(opt.nb_randomset): 
            n_obj = int(scene.split('_')[-1])
            opt.inscene_objects = n_obj
            spawn_list = np.random.choice(spawn_objects_list, opt.nb_objects, replace=False)
            spawn_list = [(f, 'medium') for f in spawn_list]
            ts.spawn_objects(spawn_list) # add random select or template load
            traj_id = 0
            while traj_id < opt.num_traj:
                #############################
                
                scene_id = {'scene': scene, 'template_id': n_set,'trajectory': traj_id, 'frame': 0}           
                success_placement = False
                while not success_placement:
                    ts.set_floor(texture_id=-1)
                    print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
                    success_placement = ts.arrange_objects(scene_id, random=True)
                    if not success_placement:
                        continue
                scene_id['frame'] += 1
                
                cnt = 0
                # 2. Move each object to a random place #
                while scene_id['frame'] < int(opt.nb_frames): #
                    if cnt > 20:
                        break
                    print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
                    success_placement = ts.random_messup_objects(scene_id)
                    if not success_placement:
                        cnt += 1
                        continue
                    scene_id['frame'] += 1
                if cnt>20:
                    continue
                traj_id += 1                
            ts.clear()
        ts.close()
