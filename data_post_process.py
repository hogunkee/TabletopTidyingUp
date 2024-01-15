import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os

def seg_postprocess(seg):
    y,x = seg.shape
    for i in range(y):
        for j in range(x):
            min_y = max(i-1,0)
            max_y = min(i+2,y)
            min_x = max(j-1,0)
            max_x = min(j+2,x)
            max_id = np.max(seg[min_y:max_y, min_x:max_x])
            min_id = np.min(seg[min_y:max_y, min_x:max_x])
            # update seg[i,j] to nearest value between max and min
            if (seg[i,j] - min_id) > (max_id - seg[i,j]):
                seg[i,j] = max_id
            else:
                seg[i,j] = min_id
    return seg

data_root = './dataset'
print(os.listdir(data_root))
for env in os.listdir(data_root):
    data_1 = os.path.join(data_root, env)
    for traj in os.listdir(data_1):
        data_2 = os.path.join(data_1, traj)
        for obj in os.listdir(data_2):
            data_3 = os.path.join(data_2, obj)
            for data in os.listdir(data_3):
                data_4 = os.path.join(data_3, data)
                print(data_4)
                seg = np.load(data_4 + '/seg_top.npy')
                new_seg = seg_postprocess(seg)
                np.save(data_4+ '/seg_top.npy', seg)
