import numpy as np
import cv2
import networkx as nx
from networkx.drawing import nx_pydot
import algorithmx
import json
import imgkit

############visualization############

# depth image show using cv2
# cv2.imshow('image', cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
# point the center of the objects in the image using cv2.circle

def visualize_sg(obj_info):
    sg = obj_info['scene_graph']
    objects_list = obj_info['objects']
    semantic_label = obj_info['semantic_label']
    visualize_list = ['left', 'on', 'front', 'leftfront', 'rightfront']
    G = nx.DiGraph()
    for obj_id in objects_list.keys():
        G.add_nodes_from([semantic_label[obj_id]])
    
    for rel in visualize_list:
        for pair in sg[rel]:
            G.add_weighted_edges_from([(semantic_label[str(pair[0])], semantic_label[str(pair[1])], rel)])
    ground_objs = []
    server = algorithmx.http_server(port=5050)
    canvas = server.canvas()
    print(G.edges)
    poses = {}
    for obj_id in objects_list.keys():
        ground_obj = True
        for pair in sg['on']:
            if obj_id == str(pair[1]):
                ground_obj = False
                break
        for pair in sg['behind']:
            if obj_id == str(pair[1]):
                ground_obj = False
                break
        for pair in sg['leftbehind']:
            if obj_id == str(pair[1]):
                ground_obj = False
                break            
        for pair in sg['rightbehind']:
            if obj_id == str(pair[1]):
                ground_obj = False
                break
        if ground_obj:
            ground_objs.append(obj_id)
    
    for obj_id in ground_objs:
        pos, orn = obj_info['state'][obj_id]
        poses[obj_id] = (pos[1], -pos[0])    
    
    i = 0
    while len(ground_objs) < len(objects_list):
        i+= 1
        for obj_id in objects_list.keys():
            if obj_id in ground_objs:
                continue
            else:
                for pair in sg['on']:
                    if obj_id == str(pair[1]) and str(pair[0]) in ground_objs and obj_id not in ground_objs :
                        ground_objs.append(obj_id)
                        pos = poses[str(pair[0])]
                        poses[obj_id] = (pos[0]+0.01, pos[1] + 0.1)
                        break
                for pair in sg['behind']:
                    if obj_id == str(pair[1]) and str(pair[0]) in ground_objs and obj_id not in ground_objs:
                        ground_objs.append(obj_id)
                        pos = poses[str(pair[0])]
                        poses[obj_id] = (pos[0] , pos[1]+0.3)
                        break
                for pair in sg['left']:
                    if obj_id == str(pair[1]) and str(pair[0]) in ground_objs and obj_id not in ground_objs:
                        ground_objs.append(obj_id)
                        pos = poses[str(pair[0])]
                        poses[obj_id] = (pos[0] - 0.2, pos[1])
                        break
                for pair in sg['right']:
                    if obj_id == str(pair[1]) and str(pair[0]) in ground_objs and obj_id not in ground_objs:
                        ground_objs.append(obj_id)
                        pos = poses[str(pair[0])]
                        poses[obj_id] = (pos[0] + 0.2, pos[1])
                        break
                for pair in sg['leftbehind']:
                    if obj_id == str(pair[1]) and str(pair[0]) in ground_objs and obj_id not in ground_objs:
                        ground_objs.append(obj_id)
                        pos = poses[str(pair[0])]
                        poses[obj_id] = (pos[0] - 0.2, pos[1] + 0.3)
                        break
                for pair in sg['rightbehind']:
                    if obj_id == str(pair[1]) and pair[0] in ground_objs and obj_id not in ground_objs:
                        ground_objs.append(obj_id)
                        pos = poses[str(pair[0])]
                        poses[obj_id] = (pos[0] + 0.2, pos[1] + 0.3)
                        break
        if i > 100:
            break
    for obj_id in objects_list.keys():
        if obj_id not in ground_objs:
            pos, orn = obj_info['state'][obj_id]
            poses[obj_id] = (pos[1], -pos[0])
        
    print(objects_list)
    for obj, pos in poses.items():
        if -0.001 < pos[0] < 0.001:
            poses[obj] = (0, pos[1])
        if -0.001 < pos[1] < 0.001:
            poses[obj] = (pos[0], 0)
    print(poses)

    def start():
        canvas.nodes(G.nodes).add(size=(30))
        canvas.edges(G.edges).add(labels = lambda e: {0:{'text': G.edges[e]['weight']}}, directed=True)
        for obj_id in objects_list.keys():
            pos = poses[obj_id]
            canvas.node(semantic_label[obj_id]).fixed(True).pos((str(pos[0] * 1)+'cx', str(pos[1] * 1)+'cy'))
    
    canvas.onmessage('start', start)
    server.start()
    imgkit.from_file('http://localhost:5050/', 'out.jpg')

if __name__ == '__main__':
    with open('./dataset/test/00001/obj_info.json', 'r') as f:
        obj_info = json.load(f)
    visualize_sg(obj_info)