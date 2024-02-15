import copy
import json
import os
import time
import math
import random

import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import pybullet as p
import pybullet_data
from gen_sg import generate_sg
from scene_utils import get_contact_objects, get_rotation, get_velocity
from scene_utils import cal_distance, check_on_table, generate_scene_random, generate_scene_shape, get_init_euler, get_random_pos_from_grid, get_random_pos_orn, move_object, pickable_objects_list, quaternion_multiply, random_pos_on_table

from utilities import Models, setup_sisbot, setup_sisbot_force, Camera
from collect_template_list import cat_to_name_test, cat_to_name_train, cat_to_name_inference
from graspnet_baseline.grasp_infer import GraspNetInfer

class FailToReachTargetError(RuntimeError):
    pass


class TableTopTidyingUpEnv:
    # OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.15
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.2
    GRASP_POINT_OFFSET_Z = 1.231 - 1.1

    GRASP_SUCCESS_REWARD = 1
    GRASP_FAIL_REWARD = -0.3
    PUSH_SUCCESS_REWARD = 0.5
    PUSH_FAIL_REWARD = -0.3
    DEPTH_CHANGE_THRESHOLD = 0.01
    DEPTH_CHANGE_COUNTER_THRESHOLD = 1000
    
    TABLE_HEIGHT = 0.8

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, objects_info, camera: Camera, vis=False, num_objs=3, gripper_type='85') -> None:
        self.vis = vis
        self.num_objs = num_objs ##
        self.camera = camera
        self.objects_info = objects_info
        self.set_grid()
        self.spawned_objects = None
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.objects_list = {}
        self.table_objects_list = {}

        self.pybullet_object_path = objects_info['paths']['pybullet_object_path']
        self.ycb_object_path = objects_info['paths']['ycb_object_path']
        self.housecat_object_path = objects_info['paths']['housecat_object_path']
        self.spawn_obj_num = 0
        
        self.init_euler = get_init_euler()
        self.urdf_id_names = self.get_obj_dict()
        
        if gripper_type not in ('85', '140'):
            raise NotImplementedError('Gripper %s not implemented.' % gripper_type)
        self.gripper_type = gripper_type
        
        if self.objects_info['split']=='unseen':
            self.cat_to_name = cat_to_name_test
            for cat in self.cat_to_name.keys():
                if not self.cat_to_name[cat]:
                    self.cat_to_name[cat].append(cat_to_name_train[cat][0])
        else:
            self.cat_to_name = cat_to_name_train

        for cat in self.cat_to_name.keys():
            if cat in cat_to_name_inference:
                self.cat_to_name[cat] = cat_to_name_inference[cat]

            
        self.obj_name_to_semantic_label = {}
        self.object_name_list = []
        for cat, names in self.cat_to_name.items():
            self.object_name_list += names
            for name in names:
                n = name.split('/')[-1]
                self.obj_name_to_semantic_label[n] = cat

        self.threshold = {'pose': 0.07,
                    'rotation': 0.25, #0.15,
                    'linear': 0.003,
                    'angular': 0.03} # angular : 0.003 (too tight to random collect)

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")
        self.tableID = p.loadURDF("table/table.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071], useFixedBase=True)
        self.UR5StandID = p.loadURDF("./urdf/objects/ur5_stand.urdf",
                                     [0.9, -0.49, 0.05],
                                     p.getQuaternionFromEuler([0, 0, np.pi / 2]),
                                     globalScaling=0.7,
                                     useFixedBase=True)
        self.robotID = p.loadURDF("./urdf/ur5_robotiq_%s.urdf" % gripper_type,
                                  [0.6, 0, -0.18],  # StartPosition
                                  p.getQuaternionFromEuler([0, 0, -np.pi / 2]),  # StartOrientation
                                  useFixedBase=True,
                                  flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName =\
            setup_sisbot(p, self.robotID, gripper_type)
        self.eefID = 7  # ee_link
        # Add force sensors
        p.enableJointForceTorqueSensor(self.robotID, self.joints['left_inner_finger_pad_joint'].id)
        p.enableJointForceTorqueSensor(self.robotID, self.joints['right_inner_finger_pad_joint'].id)
        # Change the friction of the gripper
        # p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=100, spinningFriction=10, rollingFriction=10)
        # p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=100, spinningFriction=10, rollingFriction=10)
        p.changeDynamics(self.robotID, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=0.5)
        p.changeDynamics(self.robotID, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=0.5)
        # for (k, v) in self.joints.items():
        #     print(k, v)

        # Do NOT reset robot before loading objects
        self.obj_state = []
        self.reset_robot()  # Then, move back

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # Task space (Cartesian space)
        self.xin = p.addUserDebugParameter("x",  -0.2,0.4, 0.11)
        self.yin = p.addUserDebugParameter("y", -0.4, 0.4, -0.11)
        self.zin = p.addUserDebugParameter("z", 0.72, 1.12, 1.0)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)  # -1.57 yaw
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        self.yawId = p.addUserDebugParameter("yaw", 0 , 6.28, 3.14)  # -3.14 pitch
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.085)

        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.085)
        self.ee_position_limit = ((-0.2, 0.4),
                                  (-0.4, 0.4),
                                  (0.72, 1.12))
        self.yaw_limit = (0.001, 6.28)
        # Observation buffer
        self.prev_observation = tuple()
        
        self.grasp_detector = GraspNetInfer()
        

    def set_grid(self):
        x = np.linspace(-8, -2, 10)
        y = np.linspace(-8, -2, 10)
        xx, yy = np.meshgrid(x, y, sparse=False)
        self.xx = xx.reshape(-1)
        self.yy = yy.reshape(-1)

    def get_obj_dict(self):
        # lets create a bunch of objects 
        pybullet_object_path = self.pybullet_object_path
        pybullet_object_names = sorted([m for m in os.listdir(pybullet_object_path) \
                            if os.path.isdir(os.path.join(pybullet_object_path, m))])
        if '__pycache__' in pybullet_object_names:
            pybullet_object_names.remove('__pycache__')
        ycb_object_path = self.ycb_object_path
        ycb_object_names = sorted([m for m in os.listdir(ycb_object_path) \
                            if os.path.isdir(os.path.join(ycb_object_path, m))])
        # exclusion_list = ['047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane', '033_spatula', '39_key']
        exclusion_list = ['039_key', '046_plastic_bolt', '047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane']
        for eo in exclusion_list:
            if eo in ycb_object_names:
                ycb_object_names.remove(eo)
        # ig_object_path = self.ig_object_path
        # ig_object_names = []
        # for m1 in os.listdir(ig_object_path):
        #     for m2 in os.listdir(os.path.join(ig_object_path, m1)):
        #         ig_object_names.append(os.path.join(m1, m2))
        housecat_object_path = self.housecat_object_path
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

        urdf_id_names = dict(zip(
            pybullet_ids + ycb_ids + housecat_ids,
            pybullet_object_names + ycb_object_names + housecat_object_names
            ))

        print('-'*40)
        print(len(urdf_id_names), 'objects can be loaded.')
        print('-'*40)
        return urdf_id_names

    def spawn_objects(self, obj_list):  # spawn_obj_list : [(obj_name, size), ...]
        print(obj_list)
        spawn_obj_list = []
        for obj_cat, size in obj_list:
            obj_name = np.random.choice(self.cat_to_name[obj_cat])  
            spawn_obj_list.append((obj_name, size))


        self.spawned_objects = copy.deepcopy(spawn_obj_list)
        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        pybullet_ids = []
        self.base_rot = {}
        self.base_size = {}
        for obj_, size in spawn_obj_list:
            object_type = obj_.split('/')[0]
            obj = obj_.split('/')[-1]
            urdf_id = urdf_ids[obj_names.index(obj)]
            # object_type = urdf_id.split('-')[0]
            if object_type=='pybullet':
                urdf_path = os.path.join(self.pybullet_object_path, obj, 'model.urdf')
            elif object_type=='ycb':
                urdf_path = os.path.join(self.ycb_object_path, obj, 'poisson', 'model.urdf')
                if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                    urdf_path = os.path.join(self.ycb_object_path, obj, 'tsdf', 'model.urdf')
                else:
                    urdf_path = os.path.join(self.ycb_object_path, obj, 'poisson', 'model.urdf')
            # elif object_type=='ig':
            #     urdf_path = os.path.join(self.ig_object_path, obj, obj.split('/')[-1]+'.urdf')
            elif object_type=='housecat':
                urdf_path = os.path.join(self.housecat_object_path, obj.split('-')[0], obj+'.urdf')

            roll, pitch, yaw = 0, 0, 0
            scale = 1
            if urdf_id in self.init_euler:
                roll, pitch, yaw, scale = np.array(self.init_euler[urdf_id])
                if size == 'large': scale = scale * 1.1
                elif size == 'small': scale = scale * 0.9
                roll, pitch, yaw = roll * np.pi / 2, pitch * np.pi / 2, yaw * np.pi / 2
            rot = get_rotation(roll, pitch, yaw)
            
            obj_id = p.loadURDF(urdf_path, [self.xx[self.spawn_obj_num], self.yy[self.spawn_obj_num], 0.15], rot, globalScaling=scale) #5.
            p.changeDynamics(obj_id, -1, rollingFriction=0.01, spinningFriction=0.01, restitution=0.01, lateralFriction=0.5)
            for i in range(100):
                p.stepSimulation()
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.base_rot[obj_id] = orn
            
            posa, posb = p.getAABB(obj_id)
            # to check object partially on other objects
            self.base_size[obj_id] = (np.abs(posa[0] - posb[0]), np.abs(posa[1] - posb[1]), np.abs(posa[2] - posb[2])) 
            
            pybullet_ids.append(obj_id)
            self.spawn_obj_num += 1
            self.objects_list[int(obj_id)] = (obj, size)

        self.current_pybullet_ids = copy.deepcopy(pybullet_ids)

    def arrange_objects(self, random=False ):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        # set objects #
        count_scene_trials = 0
        selected_objects = pybullet_ids

        while True:
            #hide objects placed on the table 
            for obj_id in self.pre_selected_objects:
                pos_hidden = [self.xx[pybullet_ids.index(obj_id)], self.yy[pybullet_ids.index(obj_id)], 0.15 ]
                p.resetBasePositionAndOrientation(obj_id, pos_hidden, self.base_rot[obj_id])

            # generate scene in a 'line' shape #
            if random:  # undefined position and rotation
                init_positions, init_rotations = generate_scene_random(len(pybullet_ids))
            else:     
                # TODO : load template?
                pass
            
            is_feasible, data_objects_list = self.load_obj_without_template(selected_objects, init_positions, init_rotations)
            
            count_scene_trials += 1
            if is_feasible or count_scene_trials > 5:
                break

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        # if failed to place objects robustly #
        if not is_feasible:
            return False
        
        self.pickable_objects = []
        self.get_obj_infos( data_objects_list) ##
        self.table_objects_list = data_objects_list
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
                rot = quaternion_multiply(init_rotations[idx], self.base_rot[obj_id])

            move_object(obj_id, pos_sel, rot)        

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        init_rotations = np.array(init_rotations)

        is_feasible = self.feasibility_check(selected_objects, init_positions, init_rotations) if not random else self.stable_check(selected_objects)
        data_objects_list = {id: self.objects_list[id] for id in selected_objects}
        is_on_table = check_on_table(data_objects_list)
        is_leaning_obj = self.leaning_check(selected_objects)
        is_feasible = is_feasible and is_on_table and not is_leaning_obj

        return is_feasible, data_objects_list

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


    def get_obj_infos(self,  objects_list):
        """Save data to a file."""
        obj_info = {}
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
        self.obj_info = obj_info

        return



    def step_simulation(self, delay=True):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis and delay is True:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def check_depth_change(self, cur_depth):
        _, prev_depth, _ = self.prev_observation
        changed_depth = cur_depth - prev_depth
        changed_depth_counter = np.sum(np.abs(changed_depth) > self.DEPTH_CHANGE_THRESHOLD)
        print('changed depth pixel count:', changed_depth_counter)
        return changed_depth_counter > self.DEPTH_CHANGE_COUNTER_THRESHOLD

    def step(self, target_obj, target_position,rot_angle, debug=False): 
        '''
        rot angle: based on the pybullet coordinate. difference between target & init. 
        target position : based on image coordinate. 0~1, 2dim
        '''
        x, y, z, roll, pitch, yaw, gripper_opening_length = self.read_debug_parameter()
        roll, pitch = 0, np.pi / 2
        orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        if debug:
            
            self.reset_robot()        
            rgb, depth, seg = self.camera.shot()
            self.move_away_arm()        
            self.move_ee((x, y, z, orn))
            
            print(self.table_objects_list)
            # plt.imshow(seg)
            # plt.show()
            self.grasp_detector.infer(rgb, depth, seg, 5)
            # self.move_gripper(gripper_opening_length, 1)
            return            

        self.reset_robot()        
        rgb, depth, seg = self.camera.shot()
        self.move_away_arm()        
        
        # graspnet needed
        # grasp_pixel = self.find_grasp_point(rgb, depth, seg, target_obj, target_position, rot_angle)
        # grasp_pixel to grasp position
        grasp_position = (x, y, z)
        grasp_angle = yaw
        grasp_world_position = grasp_position  #self.camera.rgbd_2_world(grasp_pixel[0], grasp_pixel[1], depth[grasp_pixel[1], grasp_pixel[0]])
        succeess_grasp = self.grasp(target_obj, grasp_world_position, grasp_angle)
        
        if succeess_grasp:        
            object_center = np.array([0,0]) # pixel, bbox center.
            object_center_world = self.camera.rgbd_2_world(object_center[0], object_center[1], depth[object_center[1], object_center[0]])
            target_position_world = self.camera.rgbd_2_world(target_position[0], target_position[1], depth[target_position[1], target_position[0]])
            
            displacement = np.array([grasp_world_position[0],grasp_world_position[1]]) - object_center_world
            Rot_2d = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
            height_disp = grasp_world_position[2] - self.TABLE_HEIGHT
            
            gripper_disp = Rot_2d @ displacement
            gripper_target_position = [target_position_world[0] + gripper_disp[0], target_position_world[1] + gripper_disp[1], target_position_world[2] + height_disp + 0.1]
            
            self.place(gripper_target_position, rot_angle)
            
        else:
            return False

    def grasp(self,target_obj: int, position: tuple, angle: float):
        """
        position [x y z]: The axis in real-world coordinate
        angle: float,   for grasp, it should be in [-pi/2, pi/2)
                        for push,  it should be in [0, 2pi)
        """
        x, y, z = position
        roll, pitch, yaw = 0, np.pi / 2 , angle
        orn = p.getQuaternionFromEuler([roll, pitch, yaw])

        # The return value of the step() method
        self.move_ee((x, y, self.GRIPPER_MOVING_HEIGHT, orn))  # Top-Down grasp / push

        self.open_gripper()
        self.move_ee((x, y, z + self.GRASP_POINT_OFFSET_Z, orn),
                        custom_velocity=0.05, max_step=1000)
        # item_in_gripper = self.close_gripper(check_contact=True)
        item_in_gripper = self.close_gripper(check_contact=True)
        print('Item in Gripper!')
        # When lifting the object, constantly try to close the gripper, in case of dropping
        self.move_ee((x, y, z + self.GRASP_POINT_OFFSET_Z + 0.1, orn), try_close_gripper=False,
                        custom_velocity=0.05, max_step=1000)
        # Lift 10 cm
        if item_in_gripper:
            grasped_ids = self.check_grasped_id()
            if target_obj in grasped_ids:
                print('Grasping success')
                self.move_ee((x, y, self.GRIPPER_GRASPED_LIFT_HEIGHT, orn), try_close_gripper=False, max_step=1000)
                return True
        
        return False


    def place(self):
        pass


    def reset_robot(self):
        # user_parameters = (-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
        #                    -1.5707970583733368, 0.0009377758247187636, 0.085) ##
        user_parameters = (0, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                           -1.5707970583733368, 0.0009377758247187636, 0.085) ##        
        for _ in range(100):
            for i, name in enumerate(self.controlJoints):
                if i == 6:
                    self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=user_parameters[i])
                    break
                joint = self.joints[name]
                # control robot joints
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=user_parameters[i], force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity)
                self.step_simulation(delay=False)

    def reset(self):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        # remove spawned objects before #
        self.spawned_objects = None
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.spawn_obj_num = 0
        
        self.reset_robot()

        self.move_away_arm()
        rgb, depth, seg = self.camera.shot()
        self.prev_observation = (rgb, depth, seg)
        self.reset_robot()

        return rgb, depth, seg

    def move_away_arm(self):
        joint = self.joints['shoulder_pan_joint']
        for _ in range(200):
            p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                    targetPosition=-1.57, force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
            self.step_simulation(delay=False)

    def check_grasped_id(self):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robotID, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robotID, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left + contact_right if item[2] in self.table_objects_list.keys())
        if len(contact_ids) > 1:
            print('Warning: Multiple items in hand!')
        if len(contact_ids) == 0:
            print(contact_left, contact_right)
        return list(item_id for item_id in contact_ids if item_id in self.table_objects_list.keys())

    def gripper_contact(self, bool_operator='and', force=100):
        left_index = self.joints['left_inner_finger_pad_joint'].id
        right_index = self.joints['right_inner_finger_pad_joint'].id

        contact_left = p.getContactPoints(bodyA=self.robotID, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robotID, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robotID, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robotID, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip(gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        for _ in range(step):
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)
            self.step_simulation()

    def open_gripper(self, step: int = 120):
        self.move_gripper(0.085, step)

    def close_gripper(self, step: int = 120, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        initial_position = p.getJointState(self.robotID, self.joints[self.mimicParentName].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self.gripper_contact():
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.move_gripper(current_target_open_length - 0.005)
                # print(p.getJointState(self.robotID, self.joints['left_inner_finger_pad_joint'].id))
                # self.controlGripper(stop=True)
                return True
        return False

    def move_ee(self, action, max_step=500, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix
        for _ in range(max_step):
            # apply IK
            real_xyz, real_xyzw = p.getLinkState(self.robotID, self.eefID)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            yaw = yaw + 2 * np.pi if yaw < 0 else yaw
            yaw = np.clip(yaw, *self.yaw_limit)
            orn = p.getQuaternionFromEuler([roll, pitch, yaw])
            
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            real_yaw = real_yaw + 2 * np.pi if real_yaw < 0 else real_yaw
            
            new_yaw = (yaw + real_yaw) /2
            orn_ = p.getQuaternionFromEuler([roll, pitch, new_yaw])
            joint_poses = p.calculateInverseKinematics(self.robotID, self.eefID, [x, y, z], orn_,
                                                       maxNumIterations=100, jointDamping=jd
                                                       )
            for i, name in enumerate(self.controlJoints[:-1]):  # Filter out the gripper
                joint = self.joints[name]
                pose = joint_poses[i]
                # control robot end-effector
                p.setJointMotorControl2(self.robotID, joint.id, p.POSITION_CONTROL,
                                        targetPosition=pose, force=joint.maxForce,
                                        maxVelocity=joint.maxVelocity if custom_velocity is None else custom_velocity * (i+1))

            self.step_simulation(delay=False)
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)
            # Check if contact with objects
            if check_collision_config and self.gripper_contact(**check_collision_config):
                print('Collision detected!', self.check_grasped_id())
                return False, p.getLinkState(self.robotID, self.eefID)[0:2]
            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(self.robotID, self.eefID)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            real_yaw = real_yaw + 2 * np.pi if real_yaw < 0 else real_yaw
            yaw = yaw + 2 * np.pi if yaw < 0 else yaw      
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        print('Failed to reach the target')
        return False, p.getLinkState(self.robotID, self.eefID)[0:2]

    def close(self):
        p.disconnect(self.physicsClient)
