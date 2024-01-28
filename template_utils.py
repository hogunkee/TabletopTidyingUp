import numpy as np
import pybullet as p
import json
import cv2
from gen_sg import generate_sg

def camera_top_view(w,h):   
    fov = 60
    aspect = w/h
    near = 0.02
    far = 2
    view_matrix = p.computeViewMatrix([0.0, 0.0, 1.5], [0, 0, 0.3], [-1, 0, 0])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    return view_matrix, projection_matrix

def save_data(folder_name, camera_parameters, objects_list):
    """Save data to a file."""
    obj_info = {}
    # front top  
    save_imgs(folder_name, camera_parameters)
    
    obj_semantic_label = {}
    obj_aabb = {}
    obj_adjacency = {}
    obj_state = {}
    n_obj = {}
    for id,obj_name in objects_list.items():
        n = n_obj.get(obj_name_to_semantic_label[obj_name], 0)
        n_obj[obj_name_to_semantic_label[obj_name]] = n + 1
        obj_semantic_label[id] = obj_name_to_semantic_label[obj_name] + '_' + str(n)
        obj_aabb[id] = p.getAABB(id)
        obj_state[id] = p.getBasePositionAndOrientation(id)
        overlapping_objs = p.getOverlappingObjects(obj_aabb[id][0], obj_aabb[id][1])
        obj_adjacency[id] = []
        for obj in overlapping_objs:
            obj_adjacency[id].append(obj[0])
    
    obj_info['objects'] = objects_list
    obj_info['semantic_label'] = obj_semantic_label
    obj_info['obj_aabb'] = obj_aabb
    obj_info['distances'] = cal_distance(objects_list)
    obj_info['state'] = obj_state
    sg = generate_sg(obj_info)
    obj_info['pickable_objects'] = pickable_objects_list(objects_list, sg)
    obj_info['scene_graph'] = sg
    
    with open(folder_name+"/obj_info.json", "w") as f:
        json.dump(obj_info, f)

    
    return

def save_imgs(folder_name, camera_parameters):
    far = 2.0
    near = 0.02
    for view, params in camera_parameters.items():
        cam_width, cam_height, view_matrix, projection_matrix = params
        images = p.getCameraImage(cam_width,
                        cam_height,
                        view_matrix,
                        projection_matrix,
                        shadow=True,
                        renderer=p.ER_TINY_RENDERER)
        rgb = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)
        depth = far * near / (far - (far - near) * images[3])
        cv2.imwrite(folder_name+ "/" + view + "_rgb.png", rgb)
        np.save(folder_name+ "/" + view + "_depth.npy", depth)
        cv2.imwrite(folder_name+ "/" + view + "_seg.png", images[4])

def load_urdf_model(model_name, position, orientation):
    """Load model from URDF file."""
    id = p.loadURDF("./pybullet-URDF-models/urdf_models/models/"+model_name+"/model.urdf", basePosition=position, baseOrientation=orientation)
    return id

def cal_distance(objects_list):
    dist = {}
    for i in objects_list.keys():
        for j in objects_list.keys():
            if i == j:
                continue
            else:
                dist_ = p.getClosestPoints(i, j, 2.0)
                dist[str((i,j))] = 1000
                for idx in range(len(dist_)):
                    dist[str((i,j))] = min(dist_[idx][8], dist[str((i,j))])
 
    return dist

def check_on_table(objects_list):
    print("check on table")
    for obj_id in objects_list.keys():
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        if(-0.5 < pos[0] < 0.5 and -0.75 < pos[1] < 0.75 and 0.6 < pos[2] < 1.0):
            pass
        else:
            print(objects_list[obj_id], "is not on the table")
            return False 
    return True           
            
def pickable_objects_list(objects_list, sg):
    pickable_objects = []
    for obj_id in objects_list.keys():
        pickable = True
        for pair in sg['on']:
            if obj_id == pair[0]:
                pickable = False
                break
        if pickable:
            pickable_objects.append(obj_id)
            
    return pickable_objects

def move_pickable_object(obj_id, obj_info, pos, rot_angle):
    if obj_id in obj_info['pickable_objects']:
        _, orig_orn = obj_info['state'][obj_id]
        pos = [pos[0], pos[1], pos[2]+0.1]
        quat = p.getQuaternionFromEuler([0,0,rot_angle * np.pi / 180.0])
        p.resetBasePositionAndOrientation(obj_id, pos, quaternion_multiply(quat, orig_orn))
        obj_info['state'][obj_id] = p.getBasePositionAndOrientation(obj_id)
    else:
        print('*****object is not pickable******')
        exit(0)
        

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return (-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)
    
def random_object():
    return np.random.choice(list(obj_name_to_semantic_label.keys()))

