import os
import numpy as np
import pybullet as p 
import nvisii as nv
from scene_utils import update_visual_objects, get_rotation
from collect_scenes import TabletopScenes

opt = lambda : None
opt.scene_type = 'line' # 'random' or 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.dataset = 'train' #'train' or 'test'
opt.objectset = 'housecat' #'pybullet'/'ycb'/'all'
opt.pybullet_object_path = '/ssd/pybullet-URDF-models/urdf_models/models'
opt.ycb_object_path = '/ssd/YCB_dataset'
opt.ig_object_path = '/ssd/ig_dataset/objects'
opt.housecat_object_path = '/ssd/housecat6d/obj_models_small_size_final'

ts = TabletopScenes(opt)
urdf_ids = list(ts.urdf_id_names.keys())
obj_names = list(ts.urdf_id_names.values())

x = np.linspace(-0.3, 0.3, 5)
y = np.linspace(-0.4, 0.4, 5)
xx, yy = np.meshgrid(x, y, sparse=False)
xx, yy = xx.reshape(-1), yy.reshape(-1)

eye = [0.5, 0, 1.2]
at = [0, 0, 0]
camera_bird = ts.set_camera_pose(eye=eye, at=at, view='bird')
for i in range(5):
    obj_selected = sorted(obj_names)[20 * i:20 * (i+1)]
    ts.spawn_objects(obj_selected)

    for idx, obj_col_id in enumerate(ts.current_pybullet_ids):
        eye = [xx[idx]+0.3, yy[idx], 1.2]
        at = [xx[idx]-0.05, yy[idx], 0.5]
        up = (0, 0, 1)
        camera_bird.get_transform().look_at(at=at, up=up, eye=eye)

        uid = urdf_ids[idx]
        object_name = obj_names[idx]
        #object_name = ts.urdf_id_names[uid]
        object_type = uid.split('-')[0]
        while True:
            #pos_new = [0, 0, 0.7]
            pos_new = [xx[idx], yy[idx], 0.8]
            if uid in ts.init_euler:
                print('init euler:', ts.init_euler[uid])
                roll, pitch, yaw = np.array(ts.init_euler[uid]) * np.pi / 2
            else:
                print(uid, 'not in init_euler.')
                roll, pitch, yaw = 0, 0, 0
            rot_new = get_rotation(roll, pitch, yaw)
            p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot_new)
            
            for j in range(500):
                p.stepSimulation()
                if j%100==0:
                    count = len(os.listdir('render/'))
                    nv.ids = update_visual_objects(ts.current_pybullet_ids, "", nv.ids)
                    #nv.render(int(opt.width), int(opt.height), int(opt.spp))
                    nv.set_camera_entity(camera_bird)
                    nv.render_to_file(
                        width=int(opt.width), height=int(opt.height), 
                        samples_per_pixel=int(opt.spp),
                        file_path=f"render/rgb_{count:05}.png"
                    )

            x = input("Set new euler values or press OK to move on to the next object.\n")
            if x.lower()=="x":
                exit()
            elif x.lower()=="ok":
                break
            else:
                if len(x.split(','))==3:
                    euler = [float(e) for e in x.split(',')]
                    ts.init_euler[uid] = euler
                continue
            
    ts.clear()
ts.close()
