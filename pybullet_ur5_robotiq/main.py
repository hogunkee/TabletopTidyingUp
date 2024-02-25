import os
import time

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from custom_env import TableTopTidyingUpEnv
from utilities import YCBModels, Camera, Camera_front_top

def ur5_control():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    objects_cfg = { 'paths': {
            #'pybullet_object_path' : '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models',
            #'ycb_object_path' : '/home/wooseoko/workspace/hogun/pybullet_scene_gen/YCB_dataset',
            #'housecat_object_path' : '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/housecat6d/obj_models_small_size_final',
            'pybullet_object_path' : '/ssd/disk/pybullet-URDF-models/urdf_models/models',
            'ycb_object_path' : '/ssd/disk/YCB_dataset',
            'housecat_object_path' : '/ssd/disk/housecat6d/obj_models_small_size_final',
        },
        'split' : 'inference' #'train'
    }
    

    env = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=True, num_objs=5, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    objects = [('bowl', 'medium'), ('can_drink','medium'), ('plate','medium'), ('marker', 'medium'), ('soap_dish', 'medium')]
    

    env.reset()
    object_pybullet_ids = env.spawn_objects(objects)
    
    env.arrange_objects(random = True)


        
    
    while True:
        # select pick object
        # object place at the certain pos, orn.
        # use env.step_action to move the gripper to the object.
        target_pos = (np.random.random(2) * np.array([360, 480])-0.5).astype(int)
        obs = env.step(5, target_pos, np.pi/2)
        print(obs)
        #env.step(5,(180,240),0.6) # pos : (y,x)
        #print('step')
        
        #debugging
        # key control
        # keys = p.getKeyboardEvents()
        # # key "Z" is down and hold
        # if (122 in keys) and (keys[122] == 3):
        #     print('Grasping...')
        #     if env.close_gripper(check_contact=True):
        #         print('Grasped!')
        # # key R
        # if 114 in keys:
        #     env.open_gripper()
        # time.sleep(1 / 120.)


if __name__ == '__main__':
    ur5_control()
