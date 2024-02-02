import os
from os import path
import pandas as pd
import numpy as np
import pybullet as p 
import nvisii as nv
from scene_utils import update_visual_objects, remove_visual_objects, get_rotation
from collect_scenes import TabletopScenes
from scene_utils import get_object_categories

opt = lambda : None
opt.scene_type = 'line' # 'random' or 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.dataset = 'train' #'train' or 'test'
opt.objectset = 'housecat' #'pybullet'/'ycb'/'housecat'/'all'
#opt.pybullet_object_path = '/ssd/disk/pybullet-URDF-models/urdf_models/models'
#opt.ycb_object_path = '/ssd/disk/YCB_dataset'
#opt.housecat_object_path = '/ssd/disk/housecat6d/obj_models_small_size_final'
opt.pybullet_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
opt.ycb_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/YCB_dataset'
opt.housecat_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/housecat6d/obj_models_small_size_final'
# opt.pybullet_object_path = '/home/brain2/workspace/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
# opt.ycb_object_path = '/home/brain2/workspace/TabletopTidyingUp/ycb_dataset'
# opt.housecat_object_path = '/home/brain2/workspace/TabletopTidyingUp/housecat6d/obj_models_small_size_final'

ts = TabletopScenes(opt)
urdf_ids = list(ts.urdf_id_names.keys())
obj_names = list(ts.urdf_id_names.values())

x = np.linspace(-0.3, 0.3, 5)
y = np.linspace(-0.4, 0.4, 5)
xx, yy = np.meshgrid(x, y, sparse=False)
xx, yy = xx.reshape(-1), yy.reshape(-1)


euler_new = {}
eye = [0.5, 0, 1.2]
at = [0, 0, 0]
camera_bird = ts.set_camera_pose(eye=eye, at=at, view='bird')

obj_names = sorted(obj_names)
obj_names = [opt.objectset+'/'+obj for obj in obj_names]
obj_sizes = ['medium'] * len(obj_names)
objs = list(zip(obj_names, obj_sizes))

object_cat_to_name = get_object_categories()
# objs_to_spawn = [o for o in object_cat_to_name['spoon'] if opt.objectset in o]
# print(objs_to_spawn)
objs_to_spawn = ['housecat/cup-white_coffee_round_handle', 'housecat/cup-stanford', 'housecat/cup-white_hogermann', 'housecat/cup-mc_cafe', 'housecat/cup-new_york', 'housecat/cup-red_heart', 'housecat/cup-yellow_handle', 'housecat/cup-teal_pattern_ikea', 'housecat/cup-new_york_big', 'housecat/cup-grey_handle', 'housecat/cup-yellow_white_border']
objs_to_spawn = [(obj, 'medium') for obj in objs_to_spawn]

f = open('euler_%s_new.csv' %opt.objectset, 'a')
df = pd.read_csv('euler_%s_new.csv' %opt.objectset)
# cnt = len(df)+1
# urdf_ids = urdf_ids[cnt:]
# objs = objs[cnt:]

obj_selected = objs_to_spawn
ts.spawn_objects(obj_selected)

for idx, obj_col_id in enumerate(ts.current_pybullet_ids):
    eye = [xx[idx]+0.3, yy[idx], 1.2]
    at = [xx[idx]-0.05, yy[idx], 0.5]
    up = (0, 0, 1)
    camera_bird.get_transform().look_at(at=at, up=up, eye=eye)
    i = obj_names.index(objs_to_spawn[idx][0])
    uid = urdf_ids[i]
    print(uid)
    object_name = obj_names[i]
    print(object_name)
    object_type = uid.split('-')[0]
    print(ts.objects_list[obj_col_id])
    scale = 1
    origin_pos, origin_rot = p.getBasePositionAndOrientation(obj_col_id)
    while True:
        pos_new = [xx[idx], yy[idx], 0.7 + scale * 0.1]
        if uid in ts.init_euler:
            print('init euler:', ts.init_euler[uid])
            roll, pitch, yaw, _ = np.array(ts.init_euler[uid])
            roll, pitch, yaw = roll  * np.pi / 2, pitch * np.pi / 2, yaw * np.pi / 2
        rot_new = get_rotation(roll, pitch, yaw)
        p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot_new)
        print(rot_new)
        for j in range(500):
            p.stepSimulation()
            if j%100==0:
                pass
        # if True:
        #     count = len(os.listdir('render/'))
        #     nv.ids = update_visual_objects(ts.current_pybullet_ids, "", nv.ids, metallic_ids=ts.metallic_ids, glass_ids=ts.glass_ids)
        #     nv.set_camera_entity(camera_bird)
        #     nv.render(int(opt.width), int(opt.height), int(opt.spp))
        #     nv.render_to_file(width=int(opt.width), height=int(opt.height), 
        #         samples_per_pixel=int(opt.spp), file_path=f"render/rgb_{count:05}.png")

        # Format: roll, pitch, yaw, scale #
        x = input("Set new euler values.\n  format: roll, pitch, yaw, (scale)\nPress OK to move on to the next object.\nPress S to save the euler values.\nPress X to exit.\n")
        if x.lower()=="x":
            exit()
        elif x.lower()=="ok":
            k = int(uid.split('-')[-1])
            p.resetBasePositionAndOrientation(obj_col_id, origin_pos, origin_rot)
            break
        else:
            if len(x.split(','))==3:
                euler = [float(e) for e in x.split(',')] + [scale]
            elif len(x.split(','))==4:
                values = [float(e) for e in x.split(',')]
                euler = values[:3]
                scale = values[3]
                remove_visual_objects(nv.ids)
                ts.respawn_object(objs_to_spawn[idx], scale) 
            ts.init_euler[uid] = euler + [scale]
            continue
