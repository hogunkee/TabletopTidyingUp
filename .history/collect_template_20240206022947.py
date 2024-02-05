import os
from matplotlib import pyplot as plt
import pybullet_data
import pybullet as p
import numpy as np
import cv2
import time
from scene_utils import cal_distance, get_init_euler, get_rotation, quaternion_multiply
from template_utils import camera_top_view
import json
from gen_sg import generate_sg
from collect_template_list import scene_list, cat_to_name_train, cat_to_name_test

mouseX = 0
mouseY = 0
rgb = np.zeros((360,480,3), np.uint8)


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
        rgb_ = rgb.copy()
        img = cv2.circle(rgb_, (mouseX,mouseY), 2, (0,0,255), -1)
        cv2.imshow('image', img)
        print('object point x:{0}, y:{1}'.format(mouseX,mouseY))




class TemplateCollector():
    def __init__(self, save_folder) -> None:
        ### camera parameters (top view)
        self.cam_width, self.cam_height = 480,360
        self.view_matrix, self.projection_matrix = camera_top_view(self.cam_width,self.cam_height)
        self.far = 2.0
        self.near = 0.02
        self.fov = 60 # degree
        self.cam_z = 1.5
        # self.base_rot = {'~~~' : [0,0,0]}
        self.base_rot = [0,0,0]
        self.save_folder = save_folder
        self.template_list = []
        self.init_euler = get_init_euler()
        # self.test_scenes = ['D2', 'D8', 'D11', 'O3', 'O7', 'B2', 'B5', 'C4', 'C6', 'C12']
        self.test_scenes = ['D5', 'D8', 'D11', 'O3', 'O7', 'B2', 'B5', 'C4', 'C6', 'C12']
        
        # self.pybullet_object_path = './pybullet-URDF-models/urdf_models/models'
        self.pybullet_object_path = '/home/brain2/workspace/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
        # self.ycb_object_path = './YCB_dataset'
        self.ycb_object_path = '/home/brain2/workspace/ycb_dataset'
        self.ig_object_path = './ig_dataset/objects'
        # self.housecat_object_path = './housecat6d/obj_models_small_size_final'
        self.housecat_object_path = '/home/brain2/workspace/housecat6d/obj_models_small_size_final'

        
        self.templates = {
            'objects' : {},  # id : semantic label (category)
            'action_order' : [], # list of actions 0 : spawn, 1 : move, 2 : simulate
            'scene_graph' : None
        }
        
        p.connect(p.GUI)
        self.urdf_id_names = self.get_obj_dict('all')
        self.init_sim()
    
    def init_sim(self):
        # Start the PyBullet simulation
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1 / 240)

        planeID = p.loadURDF("plane.urdf")
        table_id = p.loadURDF("table/table2.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071])

        # start simulation
        for i in range(100):
            p.stepSimulation()
            time.sleep(1. / 240.)

    def get_obj_dict(self, objectset):
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



    def reset_scene(self):
        self.object_list = template.copy()
        self.spawned_objects = {}
        self.templates = {
            'objects' : {},
            'action_order' : [],
            'scene_graph' : None
        }        
        p.resetSimulation()
        self.init_sim()
    
    def simulate(self):
        for i in range(100):
            p.stepSimulation()
    
    def load_image(self):
        global rgb
        # load image
        images = p.getCameraImage(self.cam_width,
                        self.cam_height,
                        self.view_matrix,
                        self.projection_matrix,
                        shadow=True,
                        renderer=p.ER_TINY_RENDERER)
        rgb = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
        self.depth = self.far * self.near / (self.far - (self.far - self.near) * images[3])
        self.seg = images[4]
    
    def save_template(self,scene_name):
        for obj_id in self.spawned_objects.keys():
            p.removeBody(obj_id)
        if self.check_reproducibility():
            print('exist templates : ', self.template_list)
            while True:
                try:
                    template_num = input('template number to save : ')
                    if int(template_num) not in self.template_list:
                        self.template_list.append(int(template_num))    
                    break
                except:
                    print('wrong number. try again')
                    continue
            self.template_list.sort()
            with open(os.path.join(self.save_folder, '{}_template_{}.json'.format(scene_name,template_num)), 'w') as f:
                json.dump(self.templates, f)
            images = p.getCameraImage(self.cam_width,
                self.cam_height,
                self.view_matrix,
                self.projection_matrix,
                shadow=True,
                renderer=p.ER_TINY_RENDERER)
            rgb = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.save_folder, '{}_template_{}.png'.format(scene_name,template_num)), rgb)
            return True
        else:
            return False
    
    def check_reproducibility(self):
        self.load_template()
        self.template_scene_graph()
        print('scene graph : \n', self.templates['scene_graph'])
        print('this is what you want? (y/n). if not, reset scene')
        key = cv2.waitKey(0)
        if key == ord('y'):
            print('save template')
            return True
        else:
            print('cancel save template')
            print('reset scene')
            self.reset_scene()
            return False
    
    def load_template(self, view_opt=True):
        template_id_to_sim_id = {}
        self.load_template_objects = {}
        new_objects_list = {}
        new_actions = []
        for action in self.templates['action_order']:
            if view_opt:
                self.load_image()
                cv2.imshow("image", rgb)
                cv2.waitKey(100)
            action_type, obj_id, pos, rot = action
            if action_type == 0: # spawn
                obj_cat, size = self.templates['objects'][obj_id]
                obj_name = np.random.choice(self.cat_to_name[obj_cat])
                obj_path, urdf_id = self.get_object_path(obj_name)
                base_rot, scale = self.get_base_rot_and_size((obj_name, size), urdf_id)
                spawn_rot = quaternion_multiply(rot, base_rot)
                sim_obj_id = p.loadURDF(obj_path, pos, spawn_rot, globalScaling=scale)   
                template_id_to_sim_id[obj_id] = sim_obj_id
                self.load_template_objects[sim_obj_id] = obj_name
                new_objects_list[sim_obj_id] = (obj_cat, size)
                new_action = (0, sim_obj_id, pos, rot)
                
            elif action_type == 1:
                sim_obj_id = template_id_to_sim_id[obj_id]
                orig_pos, orig_rot = p.getBasePositionAndOrientation(sim_obj_id)
                spawn_rot = quaternion_multiply(rot, orig_rot)
                p.resetBasePositionAndOrientation(sim_obj_id, pos, spawn_rot)
                new_action = (1, sim_obj_id, pos, rot)
                
                
            elif action_type == 2:
                self.simulate()
                new_action = (2, None, None, None)
                
            new_actions.append(new_action)
    
        if view_opt:
            self.load_image()
            cv2.imshow("image", rgb)
            cv2.waitKey(100)
        
        self.templates['action_order'] = new_actions
        self.templates['objects'] = new_objects_list
        
        self.spawned_objects = {k:v[0] for k,v in self.templates['objects'].items()}

    def exist_templates(self, scene_name):
        file_list = os.listdir(self.save_folder)
        template_list = []
        for file in file_list:
            if scene_name in file:
                template_num = int(file.split('_')[-1].split('.')[0])
                if template_num not in template_list:
                    template_list.append(template_num)
        
        self.template_list = template_list
        self.template_list.sort()

    def template_scene_graph(self):
        obj_info = {}
        
        obj_aabb = {}
        new_objects_list = {}
        for id, obj_name in self.load_template_objects.items():
            id = int(id)
            new_objects_list[id] = obj_name
            obj_aabb[id] = p.getAABB(id)
        
        print('objects : ', new_objects_list)
        obj_info['objects'] = new_objects_list
        obj_info['obj_aabb'] = obj_aabb
        obj_info['distances'] = cal_distance(new_objects_list)
        sg = generate_sg(obj_info)
        
        self.templates['scene_graph'] = sg


    def pixel_to_pos(self, x, y, depth):
        y_c, x_c = self.cam_height/2, self.cam_width/2
        x_ = (x - x_c) / x_c
        y_ = (y - y_c) / y_c
        z_ = self.cam_z
        z_diff = depth[y,x]
        tan = np.tan(self.fov/2 * np.pi / 180)
        tany = y_ * tan 
        tanx = x_ * tan * self.cam_width / self.cam_height
        y_diff = tany * z_diff
        x_diff = tanx * z_diff
        return y_diff, x_diff, z_ - z_diff
    
    def get_clicked_object(self):
        obj_id = self.seg[mouseY, mouseX]
        if obj_id > 1:
            return obj_id
        else:
            return None

    def get_object_path(self, obj_):
        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        object_type = obj_.split('/')[0]
        obj = obj_.split('/')[-1]
        urdf_id = urdf_ids[obj_names.index(obj)]

        if object_type=='pybullet':
            urdf_path = os.path.join(self.pybullet_object_path, obj, 'model.urdf')
        elif object_type=='ycb':
            urdf_path = os.path.join(self.ycb_object_path, obj, 'poisson', 'model.urdf')
            if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                urdf_path = os.path.join(self.ycb_object_path, obj, 'tsdf', 'model.urdf')
            else:
                urdf_path = os.path.join(self.ycb_object_path, obj, 'poisson', 'model.urdf')
        elif object_type=='housecat':
            urdf_path = os.path.join(self.housecat_object_path, obj.split('-')[0], obj+'.urdf')

        return urdf_path, urdf_id




    def get_base_rot_and_size(self, obj, urdf_id):
        roll, pitch, yaw = 0, 0, 0
        obj_name, size = obj
        if urdf_id in self.init_euler:
            roll, pitch, yaw, scale = np.array(self.init_euler[urdf_id])
            if size == 'large': scale = scale * 1.2
            elif size == 'small': scale = scale * 0.7
            roll, pitch, yaw = roll * np.pi / 2, pitch * np.pi / 2, yaw * np.pi / 2
        rot = get_rotation(roll, pitch, yaw)

        return rot, scale


    def add_new_object(self, obj, pos):
        x,y,z = pos
        try:
            rot_angle = input('rotation (degree) : ')
            rot_angle = int(rot_angle)
        except:
            print('you entered wrong number. set rotation to 0')
            rot_angle = 0

        quat = p.getQuaternionFromEuler([0,0,-rot_angle * np.pi / 180.0])

        urdf_path, urdf_id = self.get_object_path(obj[0])
        base_rot, scale = self.get_base_rot_and_size(obj, urdf_id,)
        
        rot = quaternion_multiply(quat, base_rot)
        
        pos = [x, y, z + 0.1]
        obj_id = p.loadURDF(urdf_path, pos, rot, globalScaling=scale)
        return obj_id, pos, quat

    def remove_added_obj(self, obj_id):
        new_action_order = [x for x in self.templates['action_order'] if x[1] != obj_id]
        self.templates['action_order'] = new_action_order
        obj_cat, size = self.templates['objects'].pop(obj_id)
        self.spawned_objects.pop(obj_id)
        if size == 'large' or size == 'small':
            self.object_list.append(size + '_' + obj_cat)
        else:
            self.object_list.append(obj_cat)
        p.removeBody(obj_id)

    def collect_template(self, scene_name, template):
        global rgb
        self.spawned_objects = {}
        self.object_list = template.copy()
        self.exist_templates(scene_name)
        print('exist templates : ', self.template_list)

        if scene_name in self.test_scenes:
            self.cat_to_name = cat_to_name_test
            for cat in self.cat_to_name.keys():
                if not self.cat_to_name[cat]:
                    self.cat_to_name[cat].append(cat_to_name_train[cat][0])
        else:
            self.cat_to_name = cat_to_name_train

        clicked_object = None
        while True:
            self.load_image()
            cv2.imshow("image", rgb)
            cv2.setMouseCallback("image", draw_circle)
            cv2.imshow("image", rgb)        
            print('template : ', self.templates)
            print('remained object list : ', self.object_list)
            
            print('\n-------enter the action------')
            print('a : add object ')
            print('d : delete object.')
            print('s : save template')
            print('r : new scene (reset)')
            print('q : quit. go to next scene')
            print('m : move object')
            print('enter : simulate')
            key = cv2.waitKey(0)

            x_, y_ , z_ = self.pixel_to_pos(mouseX, mouseY, self.depth)
            clicked_object = self.get_clicked_object()
            if clicked_object is not None:
                print('\nclicked object : id-{}, cat-{} '.format(clicked_object, self.spawned_objects[clicked_object]))
            else: print('\nno object clicked')
            
            if(key == ord('a')):
                print('\nadd object')
                print('remained object list : ', self.object_list)
                if(len(self.object_list) == 0):
                    print('no object remained')
                    continue
                
                while True:
                    print('if you want to quit, enter q')
                    obj_cat = input('object category : ')
                    if(obj_cat == 'q'):
                        print('quit add object')
                        break
                    if 'large' in obj_cat or 'small' in obj_cat:
                        try:
                            size = obj_cat.split('_')[0]
                            obj_cat_spawn = '_'.join(obj_cat.split('_')[1:])
                        except:
                            print('wrong category')
                            continue
                    else: 
                        size = 'medium'
                        obj_cat_spawn = obj_cat
                    
                    if obj_cat_spawn not in self.cat_to_name.keys() or obj_cat not in self.object_list:
                        print('wrong category')
                        continue
                    
                    obj = np.random.choice(self.cat_to_name[obj_cat_spawn])
                    
                    obj_id, spawn_pos, spawn_rot = self.add_new_object((obj, size), [x_, y_, z_]) # spawn_rot은 base_rot에서 회전한 정도를 저장.
                    self.spawned_objects[obj_id] = obj_cat
                    self.object_list.remove(obj_cat)

                    action = (0, obj_id, spawn_pos, spawn_rot)
                    self.templates['objects'][obj_id] = (obj_cat_spawn, size)
                    self.templates['action_order'].append(action)
                    break
            
            elif(key == ord('d')):
                print('\ndelete object')
                if clicked_object is not None:
                    self.remove_added_obj(clicked_object)
                else:
                    print('no object selected')
                
            elif(key == ord('s')):
                print('\nsave template')
                if(len(self.object_list) != 0):
                    print('not all objects spawned.')
                    continue
                if self.save_template(scene_name):
                    print('save template. if you want to save more templates, press r to reset scene')                
                
            elif(key == ord('r')):
                print('really reset? (y/n)')
                key = cv2.waitKey(0)
                if key == ord('y'):
                    print('reset')
                    self.reset_scene()
                    continue
                else:
                    print('cancel reset')
                    continue
                                
            elif(key == ord('q')):
                print('really quit this scene? (y/n)')
                key = cv2.waitKey(0)
                if key == ord('y'):
                    print('quit')
                    self.reset_scene()                    
                    break
                else:
                    print('cancel quit')
                    continue
            
            elif(key == ord('m')):
                print('\nmove object')
                
                if(clicked_object is None):
                    print('no object selected')
                    continue
                
                while True:
                    self.load_image()
                    cv2.imshow("image", rgb)
                    print('[ : rotate counter clockwise')
                    print('] : rotate clockwise')
                    print('w : move forward')
                    print('s : move backward')
                    print('a : move left')
                    print('d : move right')
                    print('if you want to quit, enter q')
                    key = cv2.waitKey(0)
                    
                    pos, rot = p.getBasePositionAndOrientation(clicked_object)
                    
                    if(key == ord('[')):
                        print('\nrotate counter clockwise 30 degree')
                        new_pos = [pos[0], pos[1], min(pos[2]+0.15, 0.8)]
                        new_rot = quaternion_multiply(p.getQuaternionFromEuler([0,0,30*np.pi/180]),rot)
                        quat = p.getQuaternionFromEuler([0,0,30*np.pi/180])

                        
                    elif(key == ord(']')):
                        print('\nrotate clockwise')
                        new_pos = [pos[0], pos[1], min(pos[2]+0.15, 0.8)]
                        new_rot = quaternion_multiply(p.getQuaternionFromEuler([0,0,-30*np.pi/180]), rot)
                        quat = p.getQuaternionFromEuler([0,0,-30*np.pi/180])
                                            
                    elif(key == ord('w')):
                        print('\nmove forward 0.03')
                        new_pos = [pos[0]-0.03, pos[1], min(pos[2]+0.15, 0.8)]
                        quat = p.getQuaternionFromEuler([0,0,0])
                        new_rot = rot
                        
                    elif(key == ord('s')):
                        print('\nmove backward 0.03')
                        new_pos = [pos[0]+0.03, pos[1], min(pos[2]+0.15, 0.8)]
                        quat = p.getQuaternionFromEuler([0,0,0])
                        new_rot = rot
                        
                    elif(key == ord('a')):
                        print('\nmove left 0.03')
                        new_pos = [pos[0], pos[1]-0.03, min(pos[2]+0.15, 0.8)]
                        quat = p.getQuaternionFromEuler([0,0,0])
                        new_rot = rot
                        
                    elif(key == ord('d')):
                        print('\nmove right 0.03')
                        new_pos = [pos[0], pos[1]+0.03, min(pos[2]+0.15, 0.8)]
                        quat = p.getQuaternionFromEuler([0,0,0])
                        new_rot = rot
                        
                    elif(key == ord('q')):
                        print('quit move object')
                        break
                    
                    p.resetBasePositionAndOrientation(clicked_object, new_pos, new_rot)
                    self.templates['action_order'].append((1, clicked_object, new_pos, quat))
                            
            elif(key == 13): #enter
                print('\nsimulate')
                self.simulate()
                self.templates['action_order'].append((2, None, None, None))


if __name__ == "__main__":
    save_folder = './templates'
    collector = TemplateCollector(save_folder)
    
    # saved_scene_list = collector.check_saved_scene_and_templates()
    # collect_scene_names = ['D3','D4','D5','D8','D9', 'D10','O5','O6','O7', 'O13','B1','B2','B3','B8','C1','C6','C8', 'C10', 'C14']
    collect_scene_names = ['B3','B8','C1','C8', 'C10', 'C14']
    # 호건 : ['D1','D6', 'D12', 'D14', 'D15', 'O1','O4','O9', 'O11', 'O14','B6','B7','C2','C4','C5','C9', 'C11']
    # 우석 : ['D2','D7', 'D11', 'D13', 'D16','O2','O3','O8', 'O10', 'O12', 'B3','B5','C3','C7', 'C12', 'C14']
    # 나경 : ['D3','D4','D5','D8','D9', 'D10','O5','O6','O7', 'O13','B1','B2','B3','B8','C1','C6','C8', 'C10', 'C14']



    for scene_name in collect_scene_names:
        print('current scene : ', scene_name)
        template = scene_list[scene_name]
        collector.collect_template(scene_name, template)
        
    