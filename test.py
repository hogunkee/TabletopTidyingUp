import os
import pybullet_data
import pybullet as p
import time
import numpy as np
from utils import load_urdf_model, save_data, check_on_table, random_object


def camera_front_top_view(w,h):   
    fov = 60
    aspect = w/h
    near = 0.02
    far = 2
    view_matrix = p.computeViewMatrix([0.5, 0, 1.3], [0, 0, 0.3], [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    return view_matrix, projection_matrix

def camera_top_view(w,h):   
    fov = 60
    aspect = w/h
    near = 0.02
    far = 2
    view_matrix = p.computeViewMatrix([0.0, 0.0, 1.5], [0, 0, 0.3], [-1, 0, 0])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    return view_matrix, projection_matrix

def generate_test_scene():
    '''there is a  table in the scene.
    add some objects on the table.
    Add some randomness to the object initial position and orientation.
    object names are in obj_name_to_semantic_label.
    objects_list = {object_id:object_label}
    return the list of objects ids'''
    scene_complete = True
    objects_list = {}
    def add_object(obj_name, pos, angle):
        orn = p.getQuaternionFromEuler([0, 0, angle * np.pi / 180.0 ])
        obj_id = load_urdf_model(obj_name, pos, orn)
        objects_list[obj_id] = obj_name
        for i in range(100):
            p.stepSimulation()

    # ############################
    add_object('fork', [0.2,-0.2,0.65], 180)
    check_on_table(objects_list)
    
    add_object('knife', [0.2,0.2,0.65], 180)
    check_on_table(objects_list)
    
    add_object('spoon', [0.2,-0.23,0.65], 180)
    check_on_table(objects_list)
    
    add_object('knife',[0.2,0.23,0.65],180)
    check_on_table(objects_list)
    
    add_object('blue_plate', [0.2,0,0.65], 180)
    check_on_table(objects_list)
    
    add_object('round_plate_1', [0.2,0.0,0.7], 180)    
    check_on_table(objects_list)
    
    add_object('mug', [0,0.2,0.7], 180)
    check_on_table(objects_list)
    
    add_object('doraemon_plate',[0,0,0.7], 90)
    check_on_table(objects_list)
    
    add_object('plastic_peach', [0.05,-0.05,0.7], 180)
    check_on_table(objects_list)
    
    add_object('plastic_peach', [0,0.05,0.7], 180)
    check_on_table(objects_list)
    # #########################################
    
    return objects_list, scene_complete

# scene size
# y : -0.4 ~ 0.4
# x : -0.3 ~ 0.3

# Start the PyBullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setTimeStep(1 / 240)

# Add your simulation code here
# Create a table collision shape

planeID = p.loadURDF("plane.urdf")
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
table_id = p.loadURDF("table/table.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071])

# camera parameters

camera_parameters = {}

cam_width, cam_height = 480,360
view_matrix1, projection_matrix1 = camera_top_view(cam_width,cam_height)
camera_parameters['top'] = (cam_width,cam_height,view_matrix1,projection_matrix1)
view_matrix, projection_matrix = camera_front_top_view(cam_width,cam_height)
camera_parameters['front_top'] = (cam_width,cam_height,view_matrix,projection_matrix)

#scene generation
scene_complete = False
while scene_complete == False:
    objects_list, scene_complete = generate_test_scene()

for i in range(100):
    p.stepSimulation()
    time.sleep(1. / 240.)
    print(p.getAABB(table_id))
    # print("obj_id",obj_id)  
    # print(p.getAABB(obj_id))
    # print(p.getAABB(obj2_id))
    # print(p.getBodyInfo(obj_id))
# End the PyBullet simulation

save_data("./dataset/collect_test", camera_parameters, objects_list)

p.disconnect()

