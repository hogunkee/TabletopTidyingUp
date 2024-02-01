import os
from os import path
import pandas as pd
import numpy as np
import pybullet as p 
import nvisii as nv
from scene_utils import update_visual_objects, remove_visual_objects, get_rotation
from collect_scenes import TabletopScenes

opt = lambda : None
opt.scene_type = 'line' # 'random' or 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.dataset = 'train' #'train' or 'test'
opt.objectset = 'housecat' #'pybullet'/'ycb'/'housecat'/'all'
# opt.pybullet_object_path = '/ssd/pybullet-URDF-models/urdf_models/models'
# opt.ycb_object_path = '/ssd/YCB_dataset'
# opt.ig_object_path = '/ssd/ig_dataset/objects'
# opt.housecat_object_path = '/ssd/housecat6d/obj_models_small_size_final'
opt.pybullet_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
# opt.pybullet_object_path = '/home/brain2/workspace/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
# opt.ycb_object_path = '/home/brain2/workspace/TabletopTidyingUp/ycb_dataset'
opt.ycb_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/YCB_dataset'
# opt.ig_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/ig_dataset/objects'
opt.housecat_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/housecat6d/obj_models_small_size_final'
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

if path.exists('euler_%s_new.csv' %opt.objectset):
    f = open('euler_%s_new.csv' %opt.objectset, 'a')
    df = pd.read_csv('euler_%s_new.csv' %opt.objectset)
    # cnt = len(df)+1
    # urdf_ids = urdf_ids[cnt:]
    # objs = objs[cnt:]
    print(obj_names)

    for i in range(len(obj_names)//8+1):
        obj_selected = objs[8*i:8*(i+1)]
        obj_selected = [obj for obj in obj_selected if obj[0] != 'ycb/033_spatula']
        ts.spawn_objects(obj_selected)
        print(obj_selected)

        for idx, obj_col_id in enumerate(ts.current_pybullet_ids):
            if(obj_names[8*i+idx] != opt.objectset+ '/' + ts.objects_list[obj_col_id][0]):
                idx += 1
            eye = [xx[idx]+0.3, yy[idx], 1.2]
            at = [xx[idx]-0.05, yy[idx], 0.5]
            up = (0, 0, 1)
            camera_bird.get_transform().look_at(at=at, up=up, eye=eye)
            uid = urdf_ids[8*i+idx]
            print(uid)
            object_name = obj_names[8*i+idx]
            print(object_name)
            object_type = uid.split('-')[0]
            print(ts.objects_list[obj_col_id])
            scale = 1
            origin_pos, origin_rot = p.getBasePositionAndOrientation(obj_col_id)
            while True:
                #pos_new = [0, 0, 0.7]
                pos_new = [xx[idx], yy[idx], 0.7 + scale * 0.1]
                if uid in ts.init_euler:
                    print('init euler:', ts.init_euler[uid])
                    roll, pitch, yaw = np.array(ts.init_euler[uid]) * np.pi / 2
                else:
                    print(uid, 'has no default euler angles.')
                    if opt.objectset=='housecat':
                        roll, pitch, yaw = 1, 0, 0
                    else:
                        roll, pitch, yaw = 0, 0, 0
                rot_new = get_rotation(roll, pitch, yaw)
                p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot_new)
                
                for j in range(500):
                    p.stepSimulation()
                    if j%100==0:
                        pass
                        # count = len(os.listdir('render/'))
                        # nv.ids = update_visual_objects(ts.current_pybullet_ids, "", nv.ids)
                        # nv.set_camera_entity(camera_bird)
                        # nv.render(int(opt.width), int(opt.height), int(opt.spp))
                        #nv.render_to_file(width=int(opt.width), height=int(opt.height), 
                        #    samples_per_pixel=int(opt.spp), file_path=f"render/rgb_{count:05}.png")

                # Format: roll, pitch, yaw, scale #
                x = input("Set new euler values.\n  format: roll, pitch, yaw, (scale)\nPress OK to move on to the next object.\nPress S to save the euler values.\nPress X to exit.\n")
                if x.lower()=="x":
                    exit()
                elif x.lower()=="ok":
                    k = int(uid.split('-')[-1])
                    if uid not in ts.init_euler:
                        euler_new[k] = [roll, pitch, yaw]
                    else:
                        euler_new[k] = ts.init_euler[uid]
                    print('-'*40)
                    p.resetBasePositionAndOrientation(obj_col_id, origin_pos, origin_rot)
                    # save element at the same time
                    element = [k, *euler_new[k], scale]
                    element = [str(e) for e in element]
                    line = '\t'.join(element) + '\n'
                    if path.exists('euler_%s_new.csv' %opt.objectset):
                        f = open('euler_%s_new.csv' %opt.objectset, 'a')
                        f.write(line)
                        f.close()
                    else:
                        f = open('euler_%s_new.csv' %opt.objectset, 'w')
                        f.write(line)
                        f.close()
                    break
                elif x.lower()=="s":
                    # save csv file #
                    with open('euler_%s_new.csv' %opt.objectset, 'w') as f:
                        for k in euler_new:
                            elements = [k, *euler_new[k], scale]
                            elements = [str(e) for e in elements]
                            line = '\t'.join(elements) + '\n'
                            f.write(line)
                    f.close()
                    print('-'*40)
                    print('euler_%s_new.csv saved.'%opt.objectset)
                    print('-'*40)
                    p.resetBasePositionAndOrientation(obj_col_id, origin_pos, origin_rot)
                    break
                else:
                    if len(x.split(','))==3:
                        euler = [float(e) for e in x.split(',')]
                    elif len(x.split(','))==4:
                        values = [float(e) for e in x.split(',')]
                        euler = values[:3]
                        scale = values[3]
                        remove_visual_objects(nv.ids)
                        ts.respawn_object(object_name, scale) 
                    ts.init_euler[uid] = euler
                    continue
                
        ts.clear()
    ts.close()
else:
    f = open('euler_%s_new.csv' %opt.objectset, 'w')

    for i in range(len(obj_names)//8+1):
        obj_selected = objs[8*i:8*(i+1)]
        ts.spawn_objects(obj_selected)
        print(obj_selected)

        for idx, obj_col_id in enumerate(ts.current_pybullet_ids):
            eye = [xx[idx]+0.3, yy[idx], 1.2]
            at = [xx[idx]-0.05, yy[idx], 0.5]
            up = (0, 0, 1)
            camera_bird.get_transform().look_at(at=at, up=up, eye=eye)

            uid = urdf_ids[8*i+idx]
            object_name = obj_names[8*i+idx]
            print(object_name)
            object_type = uid.split('-')[0]
            scale = 1
            origin_pos, origin_rot = p.getBasePositionAndOrientation(obj_col_id)
            while True:
                #pos_new = [0, 0, 0.7]
                pos_new = [xx[idx], yy[idx], 0.7 + scale * 0.1]
                if uid in ts.init_euler:
                    print('init euler:', ts.init_euler[uid])
                    roll, pitch, yaw = np.array(ts.init_euler[uid]) * np.pi / 2
                else:
                    print(uid, 'has no default euler angles.')
                    if opt.objectset=='housecat':
                        roll, pitch, yaw = 1, 0, 0
                    else:
                        roll, pitch, yaw = 0, 0, 0
                rot_new = get_rotation(roll, pitch, yaw)
                p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot_new)
                
                for j in range(500):
                    p.stepSimulation()
                    if j%100==0:
                        pass
                        # count = len(os.listdir('render/'))
                        # nv.ids = update_visual_objects(ts.current_pybullet_ids, "", nv.ids)
                        # nv.set_camera_entity(camera_bird)
                        # nv.render(int(opt.width), int(opt.height), int(opt.spp))
                        #nv.render_to_file(width=int(opt.width), height=int(opt.height), 
                        #    samples_per_pixel=int(opt.spp), file_path=f"render/rgb_{count:05}.png")

                # Format: roll, pitch, yaw, scale #
                x = input("Set new euler values.\n  format: roll, pitch, yaw, (scale)\nPress OK to move on to the next object.\nPress S to save the euler values.\nPress X to exit.\n")
                if x.lower()=="x":
                    exit()
                elif x.lower()=="ok":
                    k = int(uid.split('-')[-1])
                    if uid not in ts.init_euler:
                        euler_new[k] = [roll, pitch, yaw]
                    else:
                        euler_new[k] = ts.init_euler[uid]
                    print('-'*40)
                    p.resetBasePositionAndOrientation(obj_col_id, origin_pos, origin_rot)
                    # save element at the same time
                    element = [k, *euler_new[k], scale]
                    element = [str(e) for e in element]
                    line = '\t'.join(element) + '\n'
                    if path.exists('euler_%s_new.csv' %opt.objectset):
                        f = open('euler_%s_new.csv' %opt.objectset, 'a')
                        f.write(line)
                        f.close()
                    else:
                        f = open('euler_%s_new.csv' %opt.objectset, 'w')
                        f.write(line)
                        f.close()
                    break
                elif x.lower()=="s":
                    # save csv file #
                    with open('euler_%s_new.csv' %opt.objectset, 'w') as f:
                        for k in euler_new:
                            elements = [k, *euler_new[k], scale]
                            elements = [str(e) for e in elements]
                            line = '\t'.join(elements) + '\n'
                            f.write(line)
                    f.close()
                    print('-'*40)
                    print('euler_%s_new.csv saved.'%opt.objectset)
                    print('-'*40)
                    p.resetBasePositionAndOrientation(obj_col_id, origin_pos, origin_rot)
                    break
                else:
                    if len(x.split(','))==3:
                        euler = [float(e) for e in x.split(',')]
                    elif len(x.split(','))==4:
                        values = [float(e) for e in x.split(',')]
                        euler = values[:3]
                        scale = values[3]
                        remove_visual_objects(nv.ids)
                        ts.respawn_object(object_name, scale) 
                    ts.init_euler[uid] = euler
                    continue
                
        ts.clear()
    ts.close()
