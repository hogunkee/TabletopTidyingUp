import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
from scipy.spatial.transform import Rotation as R

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
#from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


class GraspNetInfer:
    def __init__(self):
        self.checkpoint_path = ROOT_DIR + '/checkpoints' + '/checkpoint-rs.tar'
        self.num_point = 40000
        self.num_view = 300
        self.collision_thresh = 0.01
        self.voxel_size = 0.01
        
        self.net = self.get_net()
    
    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
    
    def infer(self, rgb, d, mask, obj_id):
        end_points, cloud = self.process_data(rgb, d, mask, obj_id)
        gg = get_grasps(self.net, end_points)
        gg_new_array = []
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
            
        rot_thres = 1
        
        for i in range(len(gg)):
            r = R.from_matrix(gg.rotation_matrices[i])
            yaw, pitch, roll = r.as_euler('zyx')
            err1 = abs(yaw - np.pi/2)**2 + abs(roll - np.pi/2)**2
            err2 = abs(yaw + np.pi/2)**2 + abs(roll - np.pi/2)**2
            err3 = abs(yaw - np.pi/2)**2 + abs(roll + np.pi/2)**2
            err4 = abs(yaw + np.pi/2)**2 + abs(roll + np.pi/2)**2
            #print(err1, err2, err3, err4)

            if err1 < rot_thres or err2 < rot_thres or err3 < rot_thres or err4 < rot_thres:
                gg_new_array.append(gg.grasp_group_array[i])            

        gg.nms()
        gg.sort_by_score()
        gg = gg[:10] #:1
        
        vis_grasps(gg, cloud)
        return gg

    def process_data(self,rgb,d,mask_, obj_id):
        # load data
        color = rgb / 255.0
        color = color.astype(np.float32)
        color = color[:,:,:3]
        depth = d 

        factor_depth = 1

        # generate cloud
        camera = CameraInfo(480., 360., 311.769, 311.769, 239.5, 179.5, factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = np.zeros_like(mask_, dtype=bool)
        mask_true = np.where(mask_ == obj_id)
        # y_max, y_min = np.max(mask_true[0]), np.min(mask_true[0])
        # x_max, x_min = np.max(mask_true[1]), np.min(mask_true[1])
        # y_max = min(y_max+15, mask_.shape[0])
        # x_max = min(x_max+15, mask_.shape[1])
        # y_min = max(y_min-15, 0)
        # x_min = max(x_min-15, 0)
        mask = mask_ == obj_id
        # mask[y_min:y_max, x_min:x_max] = True
        
        # mask = depth>0
        
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        

        # convert data
        cloud = o3d.geometry.PointCloud()
        
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg


