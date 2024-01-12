import math
import pybullet as p
import numpy as np
from scene_utils import get_contact_objects

class Object:
    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.posA = []
        self.posB = []
        self.dist_threshold0 = 0.01
        self.dist_threshold1 = 0.1
        self.dist_threshold2 = 0.3
        self.adjacency = []
        self.near1 = {}
        self.near2 = {}
        self.final_near = []
        self.contact_objs = []

    def cal_center(self):
        self.center = [np.mean([a,b]) for a,b in zip(self.posA, self.posB)]
        self.x_len = np.abs(self.posA[0] - self.posB[0])
        self.y_len = np.abs(self.posA[1] - self.posB[1])
        self.z_len = np.abs(self.posA[2] - self.posB[2])


def generate_sg(obj_info):
    """Generate scene graph from object information.
    object informations come from pybullet.
    obj 3d bounding box, distances between objects."""
    objects_list = obj_info['objects']
    obj_aabb = obj_info['obj_aabb']
    obj_distance = obj_info['distances']
    
    contacts = get_contact_objects()
    obj_instances = {}
    for obj_id, label in objects_list.items():
        obj = Object(obj_id, label)
        obj.posA = obj_aabb[obj_id][0]
        obj.posB = obj_aabb[obj_id][1]
        obj.cal_center()
        obj_instances[obj_id] = obj



    for obj in obj_instances.values():
        for obj_ in obj_instances.values():
            if obj.id != obj_.id:
                dist = obj_distance[str((obj.id, obj_.id))]
                if dist < obj.dist_threshold0:
                    obj.adjacency.append(obj_.id)
                elif dist < obj.dist_threshold1:
                    obj.near1[obj_.id] = dist
                elif dist < obj.dist_threshold2:
                    obj.near2[obj_.id] = dist
            if (obj.id, obj_.id) in contacts or (obj_.id, obj.id) in contacts:
                obj.contact_objs.append(obj_.id)
                    
    for obj in obj_instances.values():
        if obj.adjacency != []:
            obj.final_near = list(obj.adjacency)
            if len(obj.final_near) == 1:
                if obj.near1 != {}:
                    obj.final_near.append(min(obj.near1, key=obj.near1.get))
                elif obj.near2 != {}:
                    obj.final_near.append(min(obj.near2, key=obj.near2.get))
        else:
            while(len(obj.final_near) < 3):
                if obj.near1 != {}:
                    obj.final_near.append(min(obj.near1, key=obj.near1.get))
                    del obj.near1[min(obj.near1, key=obj.near1.get)]
                elif obj.near2 != {}:
                    obj.final_near.append(min(obj.near2, key=obj.near2.get))
                    del obj.near2[min(obj.near2, key=obj.near2.get)]
                else:
                    break
                
        print(obj.final_near)

    relation_dict = {}
    edges = []
    for obj_id in obj_instances.keys():
        for obj2_id in obj_instances[obj_id].final_near:
            obj1 = obj_instances[obj_id]
            obj2 = obj_instances[obj2_id]

            relation = []
            if obj2_id in obj1.adjacency:
                # check if obj_ is on the obj
                if obj1.posA[0] < obj2.posA[0] and obj1.posB[0] > obj2.posB[0] and obj1.posA[1] < obj2.posA[1] and obj1.posB[1] > obj2.posB[1]:
                    if obj1.posB[2] < obj2.posB[2] and 1 not in obj2.contact_objs:
                        relation.append('on')
                elif obj2.posA[0] < obj1.posA[0] and obj2.posB[0] > obj1.posB[0] and obj2.posA[1] < obj1.posA[1] and obj2.posB[1] > obj1.posB[1]:
                    if obj2.posB[2] < obj1.posB[2] and 1 not in obj1.contact_objs:
                        relation.append('under')               
                if relation == []:
                    if obj1.center[2] < obj2.posA[2] and obj2.center[2] > obj1.posB[2] and 1 not in obj2.contact_objs:
                        relation.append('on')
                    elif obj2.center[2] < obj1.posA[2] and obj1.center[2] > obj2.posB[2] and 1 not in obj1.contact_objs:
                        relation.append('under')
            
            else:
                if (obj1.center[2] < obj2.posA[2] and obj2.center[2] > obj1.posB[2]) or (obj2.center[2] < obj1.posA[2] and obj1.center[2] > obj2.posB[2]):
                    break                    
                if obj1.posA[0] > obj2.posB[0] + 0.01:
                    relation.append('behind')
                elif obj1.posB[0] + 0.01 < obj2.posA[0]:
                    relation.append('front')
                if obj1.posA[1] > obj2.posB[1] + 0.01:
                    relation.append('left')
                elif obj1.posB[1] + 0.01 < obj2.posA[1]:
                    relation.append('right')
            
            if relation == []:
                # left behind
                if obj1.center[0] >= obj2.center[0] and obj1.center[1] >= obj2.center[1]:
                    pos_obj1 = [obj1.center[0], obj1.center[1]]
                    pos_obj2 = [obj2.center[0], obj2.center[1]]
                    
                    if obj1.x_len > obj1.y_len * 2:
                        pos_obj1[0] = obj1.center[0] - obj1.x_len / 4 
                    elif obj1.y_len > obj1.x_len * 2:
                        pos_obj1[1] = obj1.center[1] - obj1.y_len / 4
                    if obj2.x_len > obj2.y_len * 2:
                        pos_obj2[0] = obj2.center[0] + obj2.x_len / 4
                    elif obj2.y_len > obj2.x_len * 2:
                        pos_obj2[1] = obj2.center[1] + obj2.y_len / 4
                        
                    if (obj1.x_len > obj1.y_len * 2 and obj2.x_len > obj2.y_len * 2) or (obj1.y_len > obj1.x_len * 2 and obj2.y_len > obj2.x_len * 2):
                        pos_obj1 = [obj1.center[0], obj1.center[1]]
                        pos_obj2 = [obj2.center[0], obj2.center[1]]
                    
                    x = pos_obj2[0] - pos_obj1[0]
                    y = pos_obj2[1] - pos_obj1[1]
                    theta = math.atan2(y, x)
                    if theta < -5 * math.pi / 6 or theta > 5 * math.pi / 6:
                        relation.append('behind')
                    elif theta >= -5 * math.pi / 6 and theta < - 2 * math.pi / 3:
                        relation.append('left')
                        relation.append('behind')
                    elif theta >= -2 * math.pi / 3 and theta < - math.pi / 3:
                        relation.append('left')
                # left front
                if obj1.center[0] < obj2.center[0] and obj1.center[1] >= obj2.center[1]:
                    pos_obj1 = [obj1.center[0], obj1.center[1]]
                    if obj1.x_len > obj1.y_len * 2:
                        pos_obj1[0] = obj1.center[0] + obj1.x_len / 4 
                    elif obj1.y_len > obj1.x_len * 2:
                        pos_obj1[1] = obj1.center[1] - obj1.y_len / 4
                    pos_obj2 = [obj2.center[0], obj2.center[1]]
                    if obj2.x_len > obj2.y_len * 2:
                        pos_obj2[0] = obj2.center[0] - obj2.x_len / 4
                    elif obj2.y_len > obj2.x_len * 2:
                        pos_obj2[1] = obj2.center[1] + obj2.y_len / 4
                        
                    if (obj1.x_len > obj1.y_len * 2 and obj2.x_len > obj2.y_len * 2) or (obj1.y_len > obj1.x_len * 2 and obj2.y_len > obj2.x_len * 2):
                        pos_obj1 = [obj1.center[0], obj1.center[1]]
                        pos_obj2 = [obj2.center[0], obj2.center[1]]                        
                        
                    x = pos_obj2[0] - pos_obj1[0]
                    y = pos_obj2[1] - pos_obj1[1]
                    theta = math.atan2(y, x)
                    if theta >= -2 * math.pi / 3 and theta < - math.pi / 3:
                        relation.append('left')
                    elif theta >= - math.pi / 3 and theta < - math.pi / 6:
                        relation.append('front')
                        relation.append('left')
                    elif theta >= - math.pi / 6 and theta < math.pi / 6:
                        relation.append('front')
                # right behind
                if obj1.center[0] >= obj2.center[0] and obj1.center[1] < obj2.center[1]:
                    pos_obj1 = [obj1.center[0], obj1.center[1]]
                    if obj1.x_len > obj1.y_len * 2:
                        pos_obj1[0] = obj1.center[0] - obj1.x_len / 4 
                    elif obj1.y_len > obj1.x_len * 2:
                        pos_obj1[1] = obj1.center[1] + obj1.y_len / 4
                    pos_obj2 = [obj2.center[0], obj2.center[1]]
                    if obj2.x_len > obj2.y_len * 2:
                        pos_obj2[0] = obj2.center[0] + obj2.x_len / 4
                    elif obj2.y_len > obj2.x_len * 2:
                        pos_obj2[1] = obj2.center[1] - obj2.y_len / 4
                        
                    if (obj1.x_len > obj1.y_len * 2 and obj2.x_len > obj2.y_len * 2) or (obj1.y_len > obj1.x_len * 2 and obj2.y_len > obj2.x_len * 2):
                        pos_obj1 = [obj1.center[0], obj1.center[1]]
                        pos_obj2 = [obj2.center[0], obj2.center[1]]                        
                        
                    x = pos_obj2[0] - pos_obj1[0]
                    y = pos_obj2[1] - pos_obj1[1]
                    theta = math.atan2(y, x)
                    if theta < -5 * math.pi / 6 or theta > 5 * math.pi / 6:
                        relation.append('behind')
                    elif theta <= 5 * math.pi / 6 and theta > 2 * math.pi / 3:
                        relation.append('right')
                        relation.append('behind')
                    elif theta <= 2 * math.pi / 3 and theta >  math.pi / 3:
                        relation.append('right')
                # right front
                if obj1.center[0] < obj2.center[0] and obj1.center[1] < obj2.center[1]:
                    pos_obj1 = [obj1.center[0], obj1.center[1]]
                    if obj1.x_len > obj1.y_len * 2:
                        pos_obj1[0] = obj1.center[0] + obj1.x_len / 4 
                    elif obj1.y_len > obj1.x_len * 2:
                        pos_obj1[1] = obj1.center[1] + obj1.y_len / 4
                    pos_obj2 = [obj2.center[0], obj2.center[1]]
                    if obj2.x_len > obj2.y_len * 2:
                        pos_obj2[0] = obj2.center[0] - obj2.x_len / 4
                    elif obj2.y_len > obj2.x_len * 2:
                        pos_obj2[1] = obj2.center[1] - obj2.y_len / 4
                        
                    if (obj1.x_len > obj1.y_len * 2 and obj2.x_len > obj2.y_len * 2) or (obj1.y_len > obj1.x_len * 2 and obj2.y_len > obj2.x_len * 2):
                        pos_obj1 = [obj1.center[0], obj1.center[1]]
                        pos_obj2 = [obj2.center[0], obj2.center[1]]                        
                        
                    x = pos_obj2[0] - pos_obj1[0]
                    y = pos_obj2[1] - pos_obj1[1]
                    theta = math.atan2(y, x)
                    if theta <= 2 * math.pi / 3 and theta > math.pi / 3:
                        relation.append('right')
                    elif theta <=  math.pi / 3 and theta > math.pi / 6:
                        relation.append('front')
                        relation.append('right')
                    elif theta <= math.pi / 6 and theta > - math.pi / 6:
                        relation.append('front')
            
            edges.append((obj1.id, obj2.id, (relation)))
    print(edges)
    return refine_sg(edges)


def refine_sg(sg):
    # A (right) B & B (right) C & A(right)C -> remove A(right)C
    refined_sg = {}
    rel_pairs = {'left':[], 'right':[], 'front':[], 'behind':[], 'on':[], 'under':[], 'leftbehind':[], 'leftfront':[], 'rightbehind':[], 'rightfront':[]}
    reverse_relation = {'left': 'right', 'right': 'left', 'front': 'behind', 'behind': 'front', 'on': 'under', 'under': 'on', 'leftbehind': 'rightfront', 'leftfront': 'rightbehind', 'rightbehind': 'leftfront', 'rightfront': 'leftbehind'}

    for id1,id2,rel in sg:
        if rel != []:
            rel_pairs[get_rel_string(rel)].append((id1,id2))
        
    for rel, pairs in rel_pairs.items():
        for id1, id2 in pairs:
            if (id2, id1) not in rel_pairs[reverse_relation[rel]]:
                rel_pairs[reverse_relation[rel]].append((id2,id1))
    
    for rel, pairs in rel_pairs.items():
        remove_pair_list = []
        for pair1 in pairs:
            for pair2 in pairs:
                if pair1 != pair2:
                    if pair1[1] == pair2[0] and (pair1[0], pair2[1]) in pairs:
                        remove_pair_list.append((pair1[0],pair2[1]))
        rel_pairs[rel] = [pair for pair in pairs if pair not in remove_pair_list]
    print(rel_pairs)
    for rel, pairs in rel_pairs.items():
        if ''.join(rel) in ['leftbehind', 'behindleft']:
            refined_sg['leftbehind'] = pairs
        elif ''.join(rel) in ['rightbehind', 'behindright']:
            refined_sg['rightbehind'] = pairs
        elif ''.join(rel) in ['leftfront', 'frontleft']:
            refined_sg['leftfront'] = pairs                        
        elif ''.join(rel) in ['rightfront', 'frontright']:
            refined_sg['rightfront'] = pairs
        else:
            refined_sg[''.join(rel)] = pairs
    return refined_sg

def get_rel_string(rel):
    rel_string = ''.join(rel)
    if ''.join(rel) in ['leftbehind', 'behindleft']:
        rel_string = 'leftbehind'
    elif ''.join(rel) in ['rightbehind', 'behindright']:
        rel_string = 'rightbehind'
    elif ''.join(rel) in ['leftfront', 'frontleft']:
        rel_string = 'leftfront'
    elif ''.join(rel) in ['rightfront', 'frontright']:
        rel_string = 'rightfront'
    return rel_string