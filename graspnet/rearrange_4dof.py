import os
import sys

from picknplace_env import *

import torch
import torch.nn as nn
import argparse
import json

import copy
import time
import datetime
import random
import pylab

from replay_buffer import ReplayBuffer, PER
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

sys.path.append('/home/gun/Desktop/contact_graspnet/contact_graspnet')
from visualization_utils import visualize_grasps, show_image

from backgroundsubtraction_module import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(env,
        n_actions=8,
        num_trials=100,
        scenario=-1,
        use_hsv=False,
        ):

    # Camera Intrinsic #
    fovy = env.env.sim.model.cam_fovy[env.cam_id]
    img_height = env.env.camera_height

    f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
    cam_K = np.array([[f, 0, 240],
                      [0, f, 240],
                      [0, 0, 1]])

    for ne in range(num_trials):
        ep_len = 0
        episode_reward = 0.

        check_env_ready = False
        while not check_env_ready:
            (state_img, goal_img), info = env.reset(scenario=scenario)
            check_env_ready = True

        ni = len([f for f in os.listdir('data/') if f.endswith('_goal.png')])
        Image.fromarray((state_img[0] * 255).astype(np.uint8)).save('data/scenario-%d_state_0.png' % int(ni))
        Image.fromarray((goal_img[0] * 255).astype(np.uint8)).save('data/scenario-%d_goal.png' % int(ni))

        episode_done = False
        skip_ep = False
        while ep_len < max_steps and not episode_done:
            masks = get_object_masks() # object masks
            grasps, scores = env.get_grasps(rgb, depth, visualize=False)
            object_grasps = env.extract_grasps(grasps, scores, masks)

            flag_feasible_grasp = []
            flag_pick_fail = []
            for o in object_grasps:
                if len(object_grasps[o])==0:
                    print("No grasp candidates on object '%d'."%o)
                    continue
                placement, failed_to_pick = env.picknplace(object_grasps[o], R[o], t[o])
                ep_len += 1

                if placement is None:
                    print('No feasible grasps..')
                    flag_feasible_grasp.append(False)
                    flag_pick_fail.append(True)
                else:
                    flag_feasible_grasp.append(True)
                    flag_pick_fail.append(failed_to_pick)

                if ep_len >= max_steps:
                    break



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # contact-graspnet #
    parser.add_argument('--ckpt_dir', default='/home/gun/Desktop/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001', \
                        help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.0], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    args = parser.parse_args()

    ur5env = UR5Env(render=render, camera_height=camera_height, camera_width=camera_width, \
            control_freq=5, data_format='NHWC', gpu=gpu, camera_depth=True, dataset=dataset,\
            small=small, camera_name='rlview2')
    env = picknplace_env(ur5env, num_blocks=num_blocks, mov_dist=mov_dist, max_steps=max_steps, \
            threshold=threshold, reward_type=reward_type)
    env.load_contactgraspnet(args.ckpt_dir, args.arg_configs)

    evaluate(env=env, n_actions=8, num_trials=num_trials, scenario=scenario, use_hsv=False)
