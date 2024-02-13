import cv2
import imageio
import types
import time
import numpy as np
import os
file_path = os.path.dirname(os.path.abspath(__file__))

from copy import deepcopy
from matplotlib import pyplot as plt
from reward_functions import *
from transform_utils import euler2quat, quat2mat, mat2quat, mat2euler

import sys
sys.path.append('/home/gun/Desktop/contact_graspnet/contact_graspnet')
import config_utils
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class picknplace_env(object):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, threshold=0.10, \
            angle_threshold=30, reward_type='binary'):
        self.env = ur5_env 
        self.num_blocks = num_blocks
        self.mov_dist = mov_dist
        self.block_spawn_range_x = [-0.25, 0.25] #[-0.20, 0.20] #[-0.25, 0.25]
        self.block_spawn_range_y = [-0.12, 0.3] #[-0.10, 0.30] #[-0.15, 0.35]
        self.block_range_x = [-0.31, 0.31] #[-0.25, 0.25]
        self.block_range_y = [-0.18, 0.36] #[-0.15, 0.35]
        self.eef_range_x = [-0.35, 0.35]
        self.eef_range_y = [-0.22, 0.40]
        self.z_pick = 1.045
        self.z_prepick = self.z_pick + 0.12 #2.5 * self.mov_dist
        self.z_min = 1.04

        self.time_penalty = 0.02 #0.1
        self.max_steps = max_steps
        self.step_count = 0
        self.reward_type = reward_type

        self.init_pos = [0.0, -0.23, 1.4]

        self.threshold = threshold
        self.angle_threshold = angle_threshold
        self.depth_bg = np.load(os.path.join(file_path, 'depth_bg_480.npy'))
        self.cam_id = 2
        self.cam_theta = 0 * np.pi / 180
        self.set_camera()

        self.pre_selected_objects = None
        self.init_env()

    def set_camera(self, fovy=45):
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        self.cam_K = np.array([[f, 0, 240],
                          [0, f, 240],
                          [0, 0, 1]])

        x, y, z, w = self.env.sim.model.cam_quat[self.cam_id]
        cam_rotation = quat2mat([w, x, y, z])
        cam_pose = self.env.sim.model.cam_pos[self.cam_id]
        T_cam = np.eye(4)
        T_cam[:3, :3] = cam_rotation
        T_cam[:3, 3] = cam_pose
        self.T_cam = T_cam
        cam_mat = np.eye(4)
        cam_mat[:3, :3] = cam_rotation
        cam_mat[:3, 3] = - cam_rotation.dot(cam_pose)
        self.cam_mat = cam_mat

    def load_contactgraspnet(self, ckpt_dir, arg_configs):
        self.z_range = [0.2, 1.0]
        self.local_regions = False
        self.filter_grasps = False
        self.skip_border_objects = False
        self.forward_passes = 1

        global_config = config_utils.load_config(ckpt_dir, batch_size=self.forward_passes,
                                                    arg_configs=arg_configs)
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        saver = tf.train.Saver(save_relative_paths=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.grasp_estimator.load_weights(self.sess, saver, ckpt_dir, mode='test')

    def get_force(self):
        force = self.env.sim.data.sensordata
        return force

    def get_grasps(self, rgb, depth, segmap=None, visualize=True):
        depth[:20] = depth[:20, :100].mean()
        rgb[:20] = rgb[:20, :100].mean(axis=0).mean(axis=0)

        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, \
                self.cam_K, segmap=segmap, rgb=(rgb*255).astype(np.uint8), \
                skip_border_objects=self.skip_border_objects, z_range=self.z_range)

        grasps, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, \
                pc_full, pc_segments=pc_segments, local_regions=self.local_regions, \
                filter_grasps=self.filter_grasps, forward_passes=self.forward_passes)

        if visualize:
            visualize_grasps(pc_full, grasps, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
        return grasps[-1], scores[-1]

    def get_grasp_pixel(self, grasp):
        eef_grasp = grasp.copy()
        eef_grasp[:3, 3] = eef_grasp[:3, 3] + eef_grasp[:3, :3].dot(np.array([0, 0, 0.04]))
        t = self.T_cam.dot(eef_grasp)[:3, 3]
        u, v = self.projection(t)
        return u, v

    def extract_grasps(self, grasps, scores, masks):
        candidates = {}
        for grasp, score in zip(grasps, scores):
            u, v = self.get_grasp_pixel(grasp)
            for i, m in enumerate(masks):
                m_blur = cv2.blur(m, (7, 7)).astype(bool).astype(int)
                if not i in candidates:
                    candidates[i] = []
                if m_blur[v, u] != 0:
                    candidates[i].append([grasp, score])
        for c in candidates:
            candidates[c].sort(key=lambda x: x[1], reverse=True)
        return candidates

    def picknplace(self, grasps, R, t):
        exist_feasible_grasp = False
        failed_to_pick = False
        for g in grasps:
            grasp = g[0]
            R_place = grasp[:3, :3].dot(R)
            if R_place[2, 2]<=0:
                continue

            exist_feasible_grasp = True
            self.pick(grasp)
            force = self.get_force()
            #print('force sensor:', force)
            if force[1] > -10. and force[4] > -10.:
                failed_to_pick = True
            placement = self.place(grasp, np.dot(grasp[:3, :3].T, R_place), t)
            break

        if exist_feasible_grasp:
            return placement, failed_to_pick
        else:
            #print('No feasible grasps..')
            return None, True

    def pick(self, grasp):
        real_grasp = grasp.copy()
        real_grasp[:3, 3] = real_grasp[:3, 3] - real_grasp[:3, :3].dot(np.array([0, 0, 0.04]))
        P = self.T_cam.dot(real_grasp)
        R = P[:3, :3]
        t = P[:3, 3]
        t[2] = max(t[2], self.z_min)
        quat = mat2quat(R)      # quat=[w,x,y,z]

        pre_grasp = grasp.copy()
        pre_grasp[:3, 3] = pre_grasp[:3, 3] - pre_grasp[:3, :3].dot(np.array([0, 0, 0.10]))
        P_pre = self.T_cam.dot(pre_grasp)

        self.env.move_to_pos(grasp=0.0)
        self.env.move_to_pos(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(t, [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(t, [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(P_pre[:3, 3] + np.array([0., 0., 0.1]), \
                                  quat=[quat[3], quat[0], quat[1], quat[2]], grasp=1.0)

    def place(self, grasp, R, t):
        real_grasp = grasp.copy()
        real_grasp[:3, 3] = real_grasp[:3, 3] - real_grasp[:3, :3].dot(np.array([0, 0, 0.04]))

        P = self.T_cam.dot(real_grasp)
        #P = self.cam_mat.dot(grasp)
        P[:3, :3] = P[:3, :3].dot(R)
        P[:3, 3] = P[:3, 3].dot(R) + t
        P[2, 3] = max(P[2, 3], self.z_min + 0.04)
        quat = mat2quat(P[:3, :3])

        P_pre = P.copy()
        P_pre[:3, 3] = P_pre[:3, 3] + np.array([0, 0, 0.1])
        #pre_place[:3, 3] = pre_place[:3, 3] - pre_place[:3, :3].dot(np.array([0, 0, 0.10]))

        self.env.move_to_pos_slow(P_pre[:3, 3] + np.array([0., 0., 0.1]), \
                                  quat=[quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos_slow(P[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=1.0)
        self.env.move_to_pos(P[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(P_pre[:3, 3], [quat[3], quat[0], quat[1], quat[2]], grasp=0.0)
        self.env.move_to_pos(grasp=0.0)
        placement = P_pre[:3, 3]
        return placement

    def projection(self, pose_3d):
        cam_K = deepcopy(self.cam_K)
        # cam_K[0, 0] = -cam_K[0, 0]
        x_world = np.ones(4)
        x_world[:3] = pose_3d
        p = cam_K.dot(self.cam_mat[:3].dot(x_world))
        u = p[0] / p[2]
        v = p[1] / p[2]
        return int(np.round(u)), int(np.round(v))

    def get_angle(self, R1, R2):
        R = np.multiply(R1.T, R2)
        cos_theta = (np.trace(R)-1)/2
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        R_ = np.multiply(R1, R2.T)
        cos_theta_ = (np.trace(R_)-1)/2
        cos_theta_ = np.clip(cos_theta_, -1, 1)
        theta_ = np.arccos(cos_theta_)
        #print(R)
        #print(R_)
        return theta * 180 / np.pi

    def get_angles(self, Rs1, Rs2):
        angles = []
        for r1, r2 in zip(Rs1, Rs2):
            theta = self.get_angle(r1, r2)
            angles.append(theta)
        return np.array(angles)

    def reset(self):
        pass

    def step(self, action):
        pass

    def init_env(self, scenario=-1):
        pass
