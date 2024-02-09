import os
import cv2

path = './dataset_template/test-unseen_obj-unseen_template'

scenes = os.listdir(path)
print(scenes)

for scene in scenes:
    scene_path = os.path.join(path, scene)
    templates = os.listdir(scene_path)
    for template in templates:
        template_path = os.path.join(scene_path, template)
        trajectories = os.listdir(template_path)
        for traj in trajectories:
            traj_path = os.path.join(template_path, traj)
            frame = '000'
            frame_path = os.path.join(traj_path, frame)
            image_path = os.path.join(frame_path, 'rgb_top.png')
            img = cv2.imread(image_path)
            data_name = './data_examples/' + scene + '-' + template + '-' + traj + '-' + frame + '.png'
            print(data_name)
            cv2.imwrite(data_name, img)