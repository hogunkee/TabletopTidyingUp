import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import ClutteredPushGrasp
from custom_env import TableTopTidyingUpEnv
from utilities import YCBModels, Camera, Camera_front_top


def heuristic_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((0, -0.5, 1.5), 0.1, 5, (320, 320), 40)

    env = ClutteredPushGrasp(ycb_models, camera, vis=True, num_objs=5, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off

    (rgb, depth, seg) = env.reset()
    step_cnt = 0
    while True:

        h_, w_ = np.unravel_index(depth.argmin(), depth.shape)
        x, y, z = camera.rgbd_2_world(w_, h_, depth[h_, w_])

        p.addUserDebugLine([x, y, 0], [x, y, z], [0, 1, 0])
        p.addUserDebugLine([x, y, z], [x, y, z+0.05], [1, 0, 0])

        (rgb, depth, seg), reward, done, info = env.step((x, y, z), 1, 'grasp')

        print('Step %d, grasp at %.2f,%.2f,%.2f, reward %f, done %s, info %s' %
              (step_cnt, x, y, z, reward, done, info))
        step_cnt += 1
        # time.sleep(3)


def user_control_demo():
    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    # camera_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    objects_info = { 'paths': {
            #'pybullet_object_path' : '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models',
            #'ycb_object_path' : '/home/wooseoko/workspace/hogun/pybullet_scene_gen/YCB_dataset',
            #'housecat_object_path' : '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/housecat6d/obj_models_small_size_final',
            'pybullet_object_path' : '/ssd/disk/pybullet-URDF-models/urdf_models/models',
            'ycb_object_path' : '/ssd/disk/YCB_dataset',
            'housecat_object_path' : '/ssd/disk/housecat6d/obj_models_small_size_final',
        },
        'split' : 'inference' #'train'
    }
    

    env = TableTopTidyingUpEnv(objects_info, camera_top, vis=True, num_objs=5, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    objects = [('fork', 'medium'), ('knife','medium'), ('plate','medium'), ('marker', 'medium'), ('soap_dish', 'medium')]
    

    env.reset()
    env.spawn_objects(objects)
    env.arrange_objects(random = True)

    ## Test new urdf models for inference ##
    if True:
        obj_info = env.obj_info
        for o in obj_info['objects']:
            # get object info
            obj_info = env.obj_info
            object_name = obj_info['objects'][o]
            size = obj_info['sizes'][o]
            label = obj_info['semantic_label'][o]
            state = obj_info['state'][o]
            obj_pose, obj_quat = state
            # find grasp orientation
            roll, pitch = 0, np.pi/2
            yaw = p.getEulerFromQuaternion(obj_quat)[2]
            orn = p.getQuaternionFromEuler([roll, pitch, yaw])
            # find grasp position
            x, y = obj_pose[:2]
            z = 0.8 #0.79
            if label.startswith('plate'):
                # plate
                if 'inference_round_plate' in object_name:
                    radius = 0.07
                elif 'inference_plate' in object_name:
                    radius = 0.12
                elif 'inference_blue_plate' in object_name:
                    radius = 0.08
                else:
                    radius = 0.07
                if size=='large':
                    radius *= 1.1
                elif size=='small':
                    radius *= 0.9
                if np.random.random() > 0.5:
                    x += radius * np.sin(yaw)
                    y -= radius * np.cos(yaw)
                else:
                    x -= radius * np.sin(yaw)
                    y += radius * np.cos(yaw)
                z += 0.01
            elif label.startswith('soap_dish'):
                # soap dish
                radius = 0.06
                if size=='large':
                    radius *= 1.1
                elif size=='small':
                    radius *= 0.9
                if np.random.random() > 0.5:
                    x += radius * np.sin(yaw)
                    y -= radius * np.cos(yaw)
                else:
                    x -= radius * np.sin(yaw)
                    y += radius * np.cos(yaw)
                z += 0.02
            # move to the pre-grasp pose
            env.move_ee((x, y, z+0.2, orn))
            env.move_ee((x, y, z+0.1, orn))
            # move to the grasp pose
            env.move_ee((x, y, z, orn), custom_velocity=0.05)
            env.close_gripper()
            # move to the pre-grasp pose
            env.move_ee((x, y, z+0.2, orn), custom_velocity=0.05)
            # move to the random place
            tx, ty = (np.random.random(2)-0.5) * 0.8
            env.move_ee((tx, ty, 1.0, orn))
            env.move_ee((tx, ty, 0.9, orn))
            # place the object
            env.open_gripper()

        
    
    while True:
        # select pick object
        # object place at the certain pos, orn.
        # use env.step_action to move the gripper to the object.
        env.step(None,None,None, True)
        print('step')
        # key control
        keys = p.getKeyboardEvents()
        # key "Z" is down and hold
        if (122 in keys) and (keys[122] == 3):
            print('Grasping...')
            if env.close_gripper(check_contact=True):
                print('Grasped!')
        # key R
        if 114 in keys:
            env.open_gripper()
        # time.sleep(1 / 120.)


if __name__ == '__main__':
    user_control_demo()
    # heuristic_demo()
