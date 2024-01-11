import numpy as np
import pybullet as p
import json
import cv2
from gen_sg import generate_sg

obj_name_to_semantic_label = {
    'blue_cup': 'cup',
    'blue_plate': 'plate',
    'blue_tea_box': 'tea_box',
    'book_1': 'book',
    'book_2': 'book',
    'book_3': 'book',
    'book_4': 'book',
    'book_5': 'book',
    'book_6': 'book',
    'book_holder_2': 'book_holder',
    'book_holder_3': 'book_holder',
    'bowl': 'bowl',
    'cleanser': 'cleanser',
    'clear_box': 'basket',
    'clear_box_1': 'basket',
    'conditioner': 'conditioner',
    'cracker_box': 'cracker_box',
    'doraemon_bowl': 'bowl',
    'doraemon_plate': 'tray',
    'extra_large_clamp': 'clamp',
    'flat_screwdriver': 'screwdriver',
    'fork': 'fork',
    'gelatin_box': 'box',
    'glue_1': 'glue',
    'glue_2': 'glue',
    'green_bowl': 'bowl',
    'green_cup': 'cup',
    'grey_plate': 'plate',
    'knife': 'knife',
    'large_clamp': 'clamp',
    'lipton_tea': 'tea_box',
    'medium_clamp': 'clamp',
    'mini_claw_hammer_1': 'hammer',
    'mug': 'mug',
    'orange_cup': 'cup',
    'phillips_screwdriver': 'screwdriver',
    'pink_tea_box': 'tea_box',
    'plastic_apple': 'apple',
    'plastic_banana': 'banana',
    'plastic_lemon': 'lemon',
    'plastic_orange': 'orange',
    'plastic_peach': 'peach',
    'plastic_pear': 'pear',
    'plastic_strawberry': 'strawberry',
    'plate': 'plate',
    'power_drill': 'drill',
    'pudding_box': 'box',
    'round_plate_1': 'plate',
    'round_plate_2': 'plate',
    'round_plate_3': 'plate',
    'round_plate_4': 'plate',
    'scissors': 'scissors',
    'shampoo': 'shampoo',
    'small_clamp': 'clamp',
    'spoon': 'spoon',
    'square_plate_1': 'square_plate',
    'square_plate_2': 'square_plate',
    'square_plate_3': 'square_plate',
    'square_plate_4': 'square_plate',
    'stapler_1': 'stapler',
    'stapler_2': 'stapler',
    'sugar_box': 'box',
    'two_color_hammer': 'hammer',
    'yellow_bowl': 'bowl',
    'yellow_cup': 'cup'
    
}

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


