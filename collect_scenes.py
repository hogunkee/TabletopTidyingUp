import json
import os 
import copy
import nvisii as nv
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import pybullet_data
import numpy as np
from matplotlib import pyplot as plt
from gen_sg import generate_sg
from transform_utils import euler2quat, mat2quat, quat2mat
from scene_utils import cal_distance, check_on_table, generate_scene_random, generate_scene_shape, get_init_euler, get_random_pos_from_grid, get_random_pos_orn, move_object, pickable_objects_list, quaternion_multiply, random_pos_on_table
from scene_utils import get_rotation, get_contact_objects, get_velocity
from scene_utils import update_visual_objects 
from scene_utils import remove_visual_objects, clear_scene

object_cat_to_name = {
	'airplane': ['ycb/072-c_toy_airplane', 'ycb/072-d_toy_airplane', 'ycb/072-e_toy_airplane', 'ycb/072-f_toy_airplane', 'ycb/072-h_toy_airplane', 'ycb/072-i_toy_airplane', 'ycb/072-j_toy_airplane', 'ycb/072-k_toy_airplane'],
	'ball': ['ycb/055_baseball', 'ycb/056_tennis_ball', 'ycb/057_racquetball', 'ycb/058_golf_ball'],
	'block': ['ycb/062_dice', 'ycb/070-a_colored_wood_blocks', 'ycb/077_rubiks_cube'],
	'book': [ 'pybullet/book_3', 'pybullet/book_4', 'pybullet/book_5'],
	'book_holder': ['pybullet/book_holder_1', 'pybullet/book_holder_2', 'pybullet/book_holder_3'],
	'bottle': ['housecat/bottle-85_alcool', 'housecat/bottle-alcool_hands', 'housecat/bottle-avene_skincare', 'housecat/bottle-baby_cleansing_water', 'housecat/bottle-cleansing_lotion_small', 'housecat/bottle-deodorant_spray', 'housecat/bottle-dettol_washing_machine', 'housecat/bottle-eres_inox', 'housecat/bottle-eres_soap', 'housecat/bottle-evian_frozen', 'housecat/bottle-evian_red', 'housecat/bottle-hg_meubeline', 'housecat/bottle-inox_cleaner', 'housecat/bottle-mouthwash_meridol', 'housecat/bottle-nivea', 'housecat/bottle-sanitizer_small_white', 'housecat/bottle-shampoo_doux', 'housecat/bottle-sloth', 'housecat/bottle-soupline', 'housecat/bottle-today', 'housecat/bottle-v8_small'],
	'bowl': ['pybullet/green_bowl', 'pybullet/doraemon_bowl', 'ycb/024_bowl'],
	'box': [ 'pybullet/cracker_box', 'ycb/003_cracker_box', 'ycb/008_pudding_box', 'ycb/009_gelatin_box', 'housecat/box-antikalk', 'housecat/box-barilla', 'housecat/box-bbq', 'housecat/box-bisolvon_sirup', 'housecat/box-colgate', 'housecat/box-corny', 'housecat/box-crunchy_muesli', 'housecat/box-ebly', 'housecat/box-egar_omega_3', 'housecat/box-garmin', 'housecat/box-gift', 'housecat/box-kleenex', 'housecat/box-koala', 'housecat/box-leibniz_choco', 'housecat/box-lieqi', 'housecat/box-mc_cafe', 'housecat/box-principe', 'housecat/box-proschmelz', 'housecat/box-special_k', 'housecat/box-sultana_yo_fruit_render', 'housecat/box-toffifee', 'housecat/box-wasabi'],
	'can': [ 'ycb/005_tomato_soup_can', 'ycb/007_tuna_fish_can', 'ycb/010_potted_meat_can', 'housecat/can-beer_delta', 'housecat/can-cheakpeas', 'housecat/can-coke_small', 'housecat/can-concentrated_tall', 'housecat/can-concentrated_tiny', 'housecat/can-corned_beef', 'housecat/can-corn_small', 'housecat/can-corn', 'housecat/can-fanta', 'housecat/can-fricadelles', 'housecat/can-hoegaarden', 'housecat/can-kidney_beans', 'housecat/can-large_tuna', 'housecat/can-meatball', 'housecat/can-monster_blue', 'housecat/can-monster', 'housecat/can-olive_anchois', 'housecat/can-redbull', 'housecat/can-sheba_cat', 'housecat/can-small_tuna', 'housecat/can-spam', 'housecat/can-target_beef', 'housecat/can-tuna_salad'],
	'clamp': ['pybullet/small_clamp', 'pybullet/medium_clamp', 'pybullet/large_clamp', 'pybullet/extra_large_clamp', 'ycb/049_small_clamp', 'ycb/050_medium_clamp', 'ycb/051_large_clamp', 'ycb/052_extra_large_clamp'],
	'cleanser': ['pybullet/magic_clean', 'ycb/021_bleach_cleanser'],
	'clear_box': ['pybullet/clear_box', 'pybullet/clear_box_1', 'pybullet/clear_box_2'],
	'conditioner': ['pybullet/conditioner'],
	'cup': ['pybullet/green_cup', 'pybullet/orange_cup', 'pybullet/mug', 'ycb/025_mug', 'ycb/065-a_cups', 'ycb/065-b_cups', 'ycb/065-c_cups', 'ycb/065-d_cups', 'ycb/065-e_cups', 'ycb/065-f_cups', 'ycb/065-g_cups', 'ycb/065-h_cups', 'ycb/065-i_cups', 'ycb/065-j_cups', 'housecat/cup-green_actys', 'housecat/cup-green_handle', 'housecat/cup-grey_handle', 'housecat/cup-mc_cafe', 'housecat/cup-new_york_big', 'housecat/cup-new_york', 'housecat/cup-plastic_green_flowers', 'housecat/cup-red_heart', 'housecat/cup-red', 'housecat/cup-stanford', 'housecat/cup-teal_pattern_ikea', 'housecat/cup-white_coffee_round_handle', 'housecat/cup-white_hogermann', 'housecat/cup-white_whisker', 'housecat/cup-yellow_handle', 'housecat/cup-yellow_white_border'],
	'drill': ['pybullet/power_drill', 'ycb/035_power_drill'],
	'etc': ['pybullet/repellent', 'pybullet/blue_moon', 'pybullet/poker_1', 'pybullet/pen_container_1', 'pybullet/pitcher', 'pybullet/orion_pie', 'pybullet/plate_holder', 'pybullet/correction_fluid', 'ycb/019_pitcher_base', 'ycb/026_sponge', 'ycb/038_padlock', 'ycb/059_chain', 'ycb/071_nine_hole_peg_test'],
	'fork': ['housecat/cutlery-fork_1_new', 'housecat/cutlery-fork_1', 'housecat/cutlery-fork_2_new', 'housecat/cutlery-fork_3_new'],
	'fruits': ['pybullet/plastic_plum', 'pybullet/plastic_lemon', 'pybullet/plastic_strawberry', 'pybullet/plastic_apple', 'pybullet/plastic_banana', 'pybullet/plastic_orange', 'ycb/011_banana', 'ycb/012_strawberry', 'ycb/013_apple', 'ycb/014_lemon', 'ycb/015_peach', 'ycb/016_pear', 'ycb/017_orange', 'ycb/018_plum'],
	'glass': ['housecat/glass-new_1', 'housecat/glass-new_2', 'housecat/glass-new_3', 'housecat/glass-new_4', 'housecat/glass-new_5', 'housecat/glass-new_6', 'housecat/glass-new_7', 'housecat/glass-new_8', 'housecat/glass-new_9', 'housecat/glass-new_10', 'housecat/glass-new_11', 'housecat/glass-new_12', 'housecat/glass-new_13', 'housecat/glass-small', 'housecat/glass-small_3', 'housecat/glass-small_4'],
	'glue': ['pybullet/glue_1', 'pybullet/glue_2'],
	'hammer': ['pybullet/two_color_hammer', 'pybullet/mini_claw_hammer_1', 'ycb/048_hammer'],
	'knife': ['housecat/cutlery-knife_1_new', 'housecat/cutlery-knife_1', 'housecat/cutlery-knife_2_new', 'housecat/cutlery-knife_2', 'housecat/cutlery-knife_3_new'],
	'lego': [ 'ycb/073-c_lego_duplo', 'ycb/073-d_lego_duplo', 'ycb/073-e_lego_duplo', 'ycb/073-f_lego_duplo', 'ycb/073-g_lego_duplo', 'ycb/073-h_lego_duplo', 'ycb/073-i_lego_duplo', 'ycb/073-j_lego_duplo', 'ycb/073-k_lego_duplo', 'ycb/073-l_lego_duplo'],
	'marbles': ['ycb/063-a_marbles', 'ycb/063-d_marbles'],
	'marker': ['pybullet/red_marker', 'pybullet/black_marker', 'pybullet/blue_marker', 'pybullet/large_marker', 'pybullet/small_marker', 'ycb/040_large_marker', 'ycb/041_small_marker'],
	'remote': ['pybullet/remote_controller_1', 'pybullet/remote_controller_2', 'housecat/remote-aircon_chunghop', 'housecat/remote-black', 'housecat/remote-comfee', 'housecat/remote-factory_svc', 'housecat/remote-grey', 'housecat/remote-heitech', 'housecat/remote-infini_fun', 'housecat/remote-japanese', 'housecat/remote-jaxster_d1170', 'housecat/remote-jaxster', 'housecat/remote-led_1', 'housecat/remote-led_2', 'housecat/remote-nvtc', 'housecat/remote-seki_care', 'housecat/remote-seki_medium', 'housecat/remote-silver', 'housecat/remote-thin_silver', 'housecat/remote-toy', 'housecat/remote-tv_white_quelle'],
	'plate': ['pybullet/round_plate_3', 'pybullet/round_plate_4', 'pybullet/plate', 'pybullet/blue_plate', 'pybullet/doraemon_plate', 'pybullet/grey_plate', 'ycb/029_plate'],
	'scissors': ['pybullet/scissors', 'ycb/037_scissors'],
	'screwdriver': ['ycb/043_phillips_screwdriver', 'ycb/044_flat_screwdriver'],
	'shampoo': ['pybullet/shampoo', 'housecat/tube-shampoo_yellow'],
	'shoe': ['housecat/shoe-blue_75_right', 'housecat/shoe-cat_grey_sandal_right', 'housecat/shoe-crocs_pink_sandal_right', 'housecat/shoe-crocs_white_cyan_right', 'housecat/shoe-crocs_yellow_sandal_right', 'housecat/shoe-fashy_white_right', 'housecat/shoe-frog_right', 'housecat/shoe-green_viva_sandal_right', 'housecat/shoe-hummel_green_sandal_right', 'housecat/shoe-magenta_holes_right', 'housecat/shoe-pink_black_sandal_right', 'housecat/shoe-pink_tiny_crocs_right', 'housecat/shoe-sky_blue_250_right', 'housecat/shoe-sky_blue_holes_right', 'housecat/shoe-skyblue_leda_right', 'housecat/shoe-sky_blue_striped_right', 'housecat/shoe-votte_sandale', 'housecat/shoe-white_black_buckles', 'housecat/shoe-white_viva_sandal_right'],
	'snack': ['pybullet/potato_chip_1', 'pybullet/potato_chip_2', 'pybullet/potato_chip_3'],
	'soap': ['pybullet/soap'],
	'soap_dish': ['pybullet/soap_dish'],
	'spoon': ['housecat/cutlery-spoon_1_new', 'housecat/cutlery-spoon_1', 'housecat/cutlery-spoon_2_new', 'housecat/cutlery-spoon_2', 'housecat/cutlery-spoon_3_new', 'housecat/cutlery-spoon_4_new', 'housecat/cutlery-spoon_5_new', 'housecat/cutlery-spoon_6_new'],
	'square_plate': ['pybullet/square_plate_1', 'pybullet/square_plate_2', 'pybullet/square_plate_3', 'pybullet/square_plate_4'],
	'stapler': ['pybullet/stapler_1', 'pybullet/stapler_2'],
	'sugar': ['pybullet/suger_1', 'pybullet/suger_2', 'pybullet/suger_3', 'pybullet/sugar_box', 'ycb/004_sugar_box'],
	'tea_box': ['pybullet/pink_tea_box', 'pybullet/lipton_tea', 'pybullet/blue_tea_box'],
	'teapot': ['housecat/teapot-ambition_brand', 'housecat/teapot-big_white_floral', 'housecat/teapot-blue_floral', 'housecat/teapot-brown_chinese', 'housecat/teapot-brown_small', 'housecat/teapot-green_grass', 'housecat/teapot-new_chinese', 'housecat/teapot-pale_pink', 'housecat/teapot-white_black_line', 'housecat/teapot-white_blue_cone_top', 'housecat/teapot-white_floral', 'housecat/teapot-white_malacasa', 'housecat/teapot-white_rectangle_sprout', 'housecat/teapot-white_rectangle', 'housecat/teapot-white_royal_norfolk', 'housecat/teapot-white_small', 'housecat/teapot-white_spherical', 'housecat/teapot-white_was_brand', 'housecat/teapot-wooden_color'],
	'toothpaste': ['pybullet/toothpaste_1', 'housecat/tube-toothpaste_signal_kids_bio'],
	'tube': ['housecat/tube-bb_cream', 'housecat/tube-cleanser_white', 'housecat/tube-creme_yellow', 'housecat/tube-hansaplast_gel', 'housecat/tube-kiwi', 'housecat/tube-mdf_filler', 'housecat/tube-perfax', 'housecat/tube-shoe_polish', 'housecat/tube-signal_pokemon', 'housecat/tube-signal', 'housecat/tube-skincare_hydralist', 'housecat/tube-skincare_provencale', 'housecat/tube-wasabi', 'housecat/tube-wipp'],
	'wrench': ['ycb/042_adjustable_wrench'],
}
# object_cat_to_name = {
#     'airplane': ['ycb/072-a_toy_airplane', 'ycb/072-b_toy_airplane'],
#     'ball': ['ycb/053_mini_soccer_ball', 'ycb/054_softball'],
#     'block': ['ycb/036_wood_block', 'ycb/061_foam_brick'],
#     'book' : ['pybullet/book_1', 'pybullet/book_2'],
#     'bottle': ['ycb/006_mustard_bottle', 'ycb/022_windex_bottle'],
#     'bowl': ['pybullet/bowl', 'pybullet/yellow_bowl'],
#     'box': ['pybullet/gelatin_box', 'pybullet/pudding_box'],
#     'can': ['ycb/001_chips_can', 'ycb/002_master_chef_can'],
#     'cleanser': ['pybullet/cleanser', 'pybullet/bleach_cleanser'],
#     'cup': ['pybullet/yellow_cup', 'pybullet/blue_cup'],
#     'drill': ['pybullet/flat_screwdriver', 'pybullet/phillips_screwdriver'],
#     'fork': ['pybullet/fork', 'ycb/030_fork'],
#     'fruits': ['pybullet/plastic_peach', 'pybullet/plastic_pear'],
#     'glass': ['ycb/023_wine_glass', 'housecat/glass-cocktail'],
#     'knife': ['pybullet/knife', 'ycb/032_knife'],
#     'lego': ['ycb/073-a_lego_duplo', 'ycb/073-b_lego_duplo'],
#     'marbles': ['ycb/063-a_marbles', 'ycb/063-b_marbles'],
#     'round_plate': ['pybullet/round_plate_1', 'pybullet/round_plate_2'],
#     'shoe': ['housecat/shoe-aqua_cyan_right', 'housecat/shoe-asifn_yellow_right'],
#     'spoon': ['pybullet/spoon', 'ycb/031_spoon', 'housecat/cutlery-spoon_1_new'],
#     'square_plate': ['pybullet/square_plate_1', 'pybullet/square_plate_2'],
#     'tube': ['housecat/tube-baby_nivea_comfort', 'housecat/tube-baby_nivea_spf']
# }
obj_name_to_semantic_label = {}
object_name_list = []
for cat, names in object_cat_to_name.items():
    object_name_list += names
    for name in names:
        n = name.split('/')[-1]
        obj_name_to_semantic_label[n] = cat

# obj_name_to_semantic_label = {
#     # pybullet-URDF-models #
#     'blue_cup': 'cup',
#     'blue_plate': 'plate',
#     'blue_tea_box': 'tea_box',
#     'book_1': 'book',
#     'book_2': 'book',
#     'book_3': 'book',
#     'book_4': 'book',
#     'book_5': 'book',
#     'book_6': 'book',
#     'book_holder_2': 'book_holder',
#     'book_holder_3': 'book_holder',
#     'bowl': 'bowl',
#     'cleanser': 'cleanser',
#     'clear_box': 'basket',
#     'clear_box_1': 'basket',
#     'conditioner': 'conditioner',
#     'cracker_box': 'cracker_box',
#     'doraemon_bowl': 'bowl',
#     'doraemon_plate': 'tray',
#     'extra_large_clamp': 'clamp',
#     'flat_screwdriver': 'screwdriver',
#     'fork': 'fork',
#     'gelatin_box': 'box',
#     'glue_1': 'glue',
#     'glue_2': 'glue',
#     'green_bowl': 'bowl',
#     'green_cup': 'cup',
#     'grey_plate': 'plate',
#     'knife': 'knife',
#     'large_clamp': 'clamp',
#     'lipton_tea': 'tea_box',
#     'medium_clamp': 'clamp',
#     'mini_claw_hammer_1': 'hammer',
#     'mug': 'mug',
#     'orange_cup': 'cup',
#     'phillips_screwdriver': 'screwdriver',
#     'pink_tea_box': 'tea_box',
#     'plastic_apple': 'apple',
#     'plastic_banana': 'banana',
#     'plastic_lemon': 'lemon',
#     'plastic_orange': 'orange',
#     'plastic_peach': 'peach',
#     'plastic_pear': 'pear',
#     'plastic_strawberry': 'strawberry',
#     'plate': 'plate',
#     'power_drill': 'drill',
#     'pudding_box': 'box',
#     'round_plate_1': 'plate',
#     'round_plate_2': 'plate',
#     'round_plate_3': 'plate',
#     'round_plate_4': 'plate',
#     'scissors': 'scissors',
#     'shampoo': 'shampoo',
#     'small_clamp': 'clamp',
#     'spoon': 'spoon',
#     'square_plate_1': 'square_plate',
#     'square_plate_2': 'square_plate',
#     'square_plate_3': 'square_plate',
#     'square_plate_4': 'square_plate',
#     'stapler_1': 'stapler',
#     'stapler_2': 'stapler',
#     'sugar_box': 'box',
#     'two_color_hammer': 'hammer',
#     'yellow_bowl': 'bowl',
#     'yellow_cup': 'cup',
    
#     # housecat6d #
#     'bottle-85_alcool': 'bottle',
#     'bottle-nivea': 'bottle', 
#     'can-fanta': 'can', 
#     'can-redbull': 'can', 
#     'cup-grey_handle': 'cup', 
#     'cup-new_york': 'cup', 
#     'cup-stanford': 'cup', 
#     'remote-black': 'remote', 
#     'remote-toy': 'remote', 
#     'teapot-blue_floral': 'teapot',
# }

class TabletopScenes(object):
    def __init__(self, opt):
        self.opt = opt

        # show an interactive window, and use "lazy" updates for faster object creation time 
        nv.initialize(headless=True, lazy_updates=True) # headless=False

        # Setup bullet physics stuff
        physicsClient = p.connect(p.GUI) # non-graphical version

        # Create a camera
        self.camera_top = None
        self.camera_front_top = None
        self.set_top_view_camera()
        self.set_front_top_view_camera()
        self.set_grid()

        self.initialize_nvisii_scene()
        self.initialize_pybullet_scene()
        self.init_euler = get_init_euler()
        self.urdf_id_names = self.get_obj_dict(opt.dataset, opt.objectset)

        self.threshold = {'pose': 0.07,
                          'rotation': 0.15,
                          'linear': 0.003,
                          'angular': 0.03} # angular : 0.003 (too tight to random collect)
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.objects_list = {}
        self.spawn_obj_num = 0

    def set_front_top_view_camera(self):
        self.camera_front_top = self.set_camera_pose(eye=(0.5, 0, 1.3), at=(0, 0, 0.3), up=(0, 0, 1), view = 'front_top')
    
    def set_top_view_camera(self):
        self.camera_top = self.set_camera_pose(eye=(0, 0, 1.45), at=(0, 0, 0.3), up=(-1, 0, 0))

    def set_camera_pose(self, eye, at=(0.1, 0, 0), up=(0, 0, 1), view='top'):
        camera = nv.entity.create(
            name = "camera_" + view,
            transform = nv.transform.create("camera_" + view),
            camera = nv.camera.create_from_fov(
                name = "camera_" + view, field_of_view = 60 * np.pi / 180,
                aspect = float(self.opt.width)/float(self.opt.height)
            ))

        camera.get_transform().look_at(at=at, up=up, eye=eye)
        # nv.set_camera_entity(camera)
        return camera


    def initialize_nvisii_scene(self):
        if not self.opt.noise is True: 
            nv.enable_denoiser()

        # Change the dome light intensity
        nv.set_dome_light_intensity(1.0)

        # atmospheric thickness makes the sky go orange, almost like a sunset
        nv.set_dome_light_sky(sun_position=(6,6,6), atmosphere_thickness=1.0, saturation=1.0)

        # add a sun light
        sun = nv.entity.create(
            name = "sun",
            mesh = nv.mesh.create_sphere("sphere"),
            transform = nv.transform.create("sun"),
            light = nv.light.create("sun")
        )
        sun.get_transform().set_position((6,6,6))
        sun.get_light().set_temperature(5780)
        sun.get_light().set_intensity(1000)

        floor = nv.entity.create(
            name="floor",
            mesh = nv.mesh.create_plane("floor"),
            transform = nv.transform.create("floor"),
            material = nv.material.create("floor")
        )
        floor.get_transform().set_position((0,0,0))
        floor.get_transform().set_scale((2, 2, 2)) #(6, 6, 6)
        floor.get_material().set_roughness(0.1)
        floor.get_material().set_base_color((0.8, 0.87, 0.88)) #(0.5,0.5,0.5)

        floor_textures = []
        texture_files = os.listdir("texture")
        texture_files = [f for f in texture_files if f.lower().endswith('.png')]

        for i, tf in enumerate(texture_files):
            tex = nv.texture.create_from_file("tex-%d"%i, os.path.join("texture/", tf))
            floor_tex = nv.texture.create_hsv("floor-%d"%i, tex, hue=0, saturation=.5, value=1.0, mix=1.0)
            floor_textures.append((tex, floor_tex))
        self.floor, self.floor_textures = floor, floor_textures
        return

    def initialize_pybullet_scene(self):
        # reset pybullet #
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setTimeStep(1 / 240)

        # set the plane and table        
        planeID = p.loadURDF("plane.urdf")
        table_id = p.loadURDF("table/table2.urdf", basePosition=[0.0,0.0,0.0], baseOrientation=[0.0,0.0,0.7071,0.7071])
        return

    def set_grid(self):
        x = np.linspace(-8, -2, 10)
        y = np.linspace(-8, -2, 10)
        xx, yy = np.meshgrid(x, y, sparse=False)
        self.xx = xx.reshape(-1)
        self.yy = yy.reshape(-1)

    def clear(self):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        # remove spawned objects before #
        remove_visual_objects(nv.ids)
        #clear_scene()
        self.spawned_objects = None
        self.pre_selected_objects = []
        self.current_pybullet_ids = []
        self.spawn_obj_num = 0
        p.resetSimulation()
        clear_scene()
        self.camera_top = None
        self.camera_front_top = None
        self.set_top_view_camera()
        self.set_front_top_view_camera()
        self.initialize_nvisii_scene()
        self.initialize_pybullet_scene()

    def close(self):
        nv.deinitialize()
        p.disconnect()

    def get_obj_dict(self, dataset, objectset):
        # lets create a bunch of objects 
        pybullet_object_path = self.opt.pybullet_object_path
        pybullet_object_names = sorted([m for m in os.listdir(pybullet_object_path) \
                            if os.path.isdir(os.path.join(pybullet_object_path, m))])
        if '__pycache__' in pybullet_object_names:
            pybullet_object_names.remove('__pycache__')
        ycb_object_path = self.opt.ycb_object_path
        ycb_object_names = sorted([m for m in os.listdir(ycb_object_path) \
                            if os.path.isdir(os.path.join(ycb_object_path, m))])
        # exclusion_list = ['047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane', '033_spatula', '39_key']
        exclusion_list = ['039_key', '046_plastic_bolt', '047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane']
        for eo in exclusion_list:
            if eo in ycb_object_names:
                ycb_object_names.remove(eo)
        # ig_object_path = self.opt.ig_object_path
        # ig_object_names = []
        # for m1 in os.listdir(ig_object_path):
        #     for m2 in os.listdir(os.path.join(ig_object_path, m1)):
        #         ig_object_names.append(os.path.join(m1, m2))
        housecat_object_path = self.opt.housecat_object_path
        housecat_categories = [h for h in os.listdir(housecat_object_path) if os.path.isdir(os.path.join(housecat_object_path, h)) and not h in ['bg', 'collision']]
        housecat_object_names = []
        for m1 in housecat_categories:
            obj_model_list = [h for h in os.listdir(os.path.join(housecat_object_path, m1)) if h.endswith('.urdf')]
            for m2 in obj_model_list:
                housecat_object_names.append(m2.split('.urdf')[0])
        housecat_object_names = sorted(housecat_object_names)

        housecat_object_names = sorted(housecat_object_names)
        
        pybullet_ids = ['pybullet-%d'%p for p in range(len(pybullet_object_names))]
        ycb_ids = ['ycb-%d'%y for y in range(len(ycb_object_names))]
        # ig_ids = ['ig-%d'%i for i in range(len(ig_object_names))]
        housecat_ids = ['housecat-%d'%h for h in range(len(housecat_object_names))]
        # urdf_id_names
        # key: {object_type}-{index} - e.g., 'pybullet-0'
        # value: {object_name} - e.g., 'black_marker'
        if objectset=='pybullet':
            urdf_id_names = dict(zip(
                pybullet_ids, 
                pybullet_object_names
                ))
        elif objectset=='ycb':
            urdf_id_names = dict(zip(
                ycb_ids, 
                ycb_object_names
                ))
        # elif objectset=='ig':
        #     urdf_id_names = dict(zip(
        #         ig_ids,
        #         ig_object_names
        #         ))
        elif objectset=='housecat':
            urdf_id_names = dict(zip(
                housecat_ids,
                housecat_object_names
                ))
        elif objectset=='all':
            urdf_id_names = dict(zip(
                pybullet_ids + ycb_ids + housecat_ids,
                pybullet_object_names + ycb_object_names + housecat_object_names
                ))

        print('-'*40)
        print(len(urdf_id_names), 'objects can be loaded.')
        print('-'*40)
        return urdf_id_names

    def spawn_objects(self, spawn_obj_list):  # spawn_obj_list : [(obj_name, size), ...]
        self.spawned_objects = copy.deepcopy(spawn_obj_list)
        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        pybullet_ids = []
        self.base_rot = {}
        self.base_size = {}
        for obj_, size in spawn_obj_list: ### TODO: 사이즈별로 (large, small) 에 따라서 크기 다르게 spawn해야함.
            object_type = obj_.split('/')[0]
            obj = obj_.split('/')[-1]
            urdf_id = urdf_ids[obj_names.index(obj)]
            # object_type = urdf_id.split('-')[0]
            if object_type=='pybullet':
                urdf_path = os.path.join(self.opt.pybullet_object_path, obj, 'model.urdf')
            elif object_type=='ycb':
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
                if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                    urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'tsdf', 'model.urdf')
                else:
                    urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
            # elif object_type=='ig':
            #     urdf_path = os.path.join(self.opt.ig_object_path, obj, obj.split('/')[-1]+'.urdf')
            elif object_type=='housecat':
                urdf_path = os.path.join(self.opt.housecat_object_path, obj.split('-')[0], obj+'.urdf')

            roll, pitch, yaw = 0, 0, 0
            if urdf_id in self.init_euler:
                roll, pitch, yaw, scale = np.array(self.init_euler[urdf_id])
                if size == 'large': scale = scale * 1.2
                elif size == 'small': scale = scale * 0.8
                roll, pitch, yaw = roll * np.pi / 2, pitch * np.pi / 2, yaw * np.pi / 2
            rot = get_rotation(roll, pitch, yaw)
            obj_id = p.loadURDF(urdf_path, [self.xx[self.spawn_obj_num], self.yy[self.spawn_obj_num], 0.15], rot, globalScaling=scale) #5.
            for i in range(100):
                p.stepSimulation()
            pos,orn = p.getBasePositionAndOrientation(obj_id)
            self.base_rot[obj_id] = orn
            
            posa, posb = p.getAABB(obj_id)
            # to check object partially on other objects
            self.base_size[obj_id] = (np.abs(posa[0] - posb[0]), np.abs(posa[1] - posb[1]), np.abs(posa[2] - posb[2])) 
            
            pybullet_ids.append(obj_id)
            self.spawn_obj_num += 1
            self.objects_list[int(obj_id)] = (obj, size)

        nv.ids = update_visual_objects(pybullet_ids, "")
        self.current_pybullet_ids = copy.deepcopy(pybullet_ids)

    def respawn_object(self, obj_, scale=1.):
        spawn_idx = self.spawned_objects.index(obj_)
        pybullet_ids = self.current_pybullet_ids
        obj_id_to_remove = pybullet_ids[spawn_idx]

        # remove the existing object
        self.base_rot.pop(obj_id_to_remove)
        self.base_size.pop(obj_id_to_remove)
        self.objects_list.pop(int(obj_id_to_remove))
        p.removeBody(obj_id_to_remove)

        urdf_ids = list(self.urdf_id_names.keys())
        obj_names = list(self.urdf_id_names.values())

        object_type = obj_[0].split('/')[0]
        obj = obj_[0].split('/')[-1]
        
        urdf_id = urdf_ids[obj_names.index(obj)]
        if object_type=='pybullet':
            urdf_path = os.path.join(self.opt.pybullet_object_path, obj, 'model.urdf')
        elif object_type=='ycb':
            urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
            if obj.startswith('022_windex_bottle') or obj.startswith('023_wine_glass') or obj.startswith('049_'):
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'tsdf', 'model.urdf')
            else:
                urdf_path = os.path.join(self.opt.ycb_object_path, obj, 'poisson', 'model.urdf')
        # elif object_type=='ig':
        #     urdf_path = os.path.join(self.opt.ig_object_path, obj, obj.split('/')[-1]+'.urdf')
        elif object_type=='housecat':
            urdf_path = os.path.join(self.opt.housecat_object_path, obj.split('-')[0], obj+'.urdf')

        roll, pitch, yaw = 0, 0, 0
        if urdf_id in self.init_euler:
            roll, pitch, yaw, _ = np.array(self.init_euler[urdf_id]) * np.pi / 2
        rot = get_rotation(roll, pitch, yaw)
        obj_id = p.loadURDF(urdf_path, [self.xx[self.spawn_obj_num], self.yy[self.spawn_obj_num], 0.15], rot, globalScaling=scale)
        for i in range(100):
            p.stepSimulation()
        pos,orn = p.getBasePositionAndOrientation(obj_id)
        self.base_rot[obj_id] = orn
        
        posa, posb = p.getAABB(obj_id)
        # to check object partially on other objects
        self.base_size[obj_id] = (np.abs(posa[0] - posb[0]), np.abs(posa[1] - posb[1]), np.abs(posa[2] - posb[2])) 
        
        pybullet_ids[spawn_idx] = obj_id
        self.objects_list[int(obj_id)] = (obj, obj_[1])

        nv.ids = update_visual_objects(pybullet_ids, "")
        self.current_pybullet_ids = copy.deepcopy(pybullet_ids)

    def set_floor(self, texture_id=-1):
        # set floor material #
        roughness = random.uniform(0.1, 0.5)
        self.floor.get_material().clear_base_color_texture()
        self.floor.get_material().set_roughness(roughness)

        if texture_id==-1: # random texture #
            f_cidx = np.random.choice(len(self.floor_textures))
            tex, floor_tex = self.floor_textures[f_cidx]
        else:
            tex, floor_tex = self.floor_textures[texture_id]
        self.floor.get_material().set_base_color_texture(floor_tex)
        self.floor.get_material().set_roughness_texture(tex)

    def arrange_objects(self, scene_id, select=True, random=False ):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        # set objects #
        count_scene_trials = 0
        selected_objects = np.random.choice(pybullet_ids, self.opt.inscene_objects, replace=False) if select else pybullet_ids

        while True:
            #hide objects placed on the table 
            for obj_id in self.pre_selected_objects:
                pos_hidden = [self.xx[pybullet_ids.index(obj_id)], self.yy[pybullet_ids.index(obj_id)], 0.15 ]
                p.resetBasePositionAndOrientation(obj_id, pos_hidden, self.base_rot[obj_id])

            # generate scene in a 'line' shape #
            if random:  # undefined position and rotation
                init_positions, init_rotations = generate_scene_random(self.opt.inscene_objects)
            else:     # defined position and rotation (rule-based : specific shape)
                init_positions, init_rotations = generate_scene_shape(self.opt.scene_type, self.opt.inscene_objects)
                # init_rotations = []                
            is_feasible, data_objects_list = self.load_obj_without_template(selected_objects, init_positions, init_rotations)
            
            count_scene_trials += 1
            if is_feasible or count_scene_trials > 5:
                break

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        # if failed to place objects robustly #
        if not is_feasible:
            return False
        
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        self.pickable_objects = []
        self.save_data(scene_id, data_objects_list)
        return True

    def load_obj_without_template(self, selected_objects, init_positions, init_rotations):
        # place new objects #
        set_base_rot = True if init_rotations == [] else False               
        for idx, obj_id in enumerate(selected_objects):
            pos_sel = init_positions[idx]
            if set_base_rot:
                rot = self.base_rot[obj_id]
                init_rotations.append(rot)
            else: # TODO: set object rotation as template's rotation or random rotation
                rot = quaternion_multiply(init_rotations[idx], self.base_rot[obj_id]) if random else init_rotations[idx]

            move_object(obj_id, pos_sel, rot)        

            # calculate distance btw objects and repose objects
            if self.opt.scene_type=='line':
                goal_dist = 0.02
                if idx==0:
                    continue
                pre_obj_id = selected_objects[idx-1]
                #pre_obj_id = copy.deepcopy(obj_id)
                objects_pair = {
                        pre_obj_id: self.objects_list[pre_obj_id][0],
                        obj_id: self.objects_list[obj_id][0]
                        }
                distance = list(cal_distance(objects_pair).values())[0]
                if distance < 0.0001:
                    continue
                # delta = dist(o1, o2) - goal_dist
                # x_hat := unit vector from o1 to o2
                # p2_new = p2 - delta * x_hat
                pre_pos_sel = init_positions[idx-1]
                x_hat = pos_sel - pre_pos_sel
                x_hat /= np.linalg.norm(x_hat)
                pos_new = pos_sel - (distance - goal_dist) * x_hat
                move_object(obj_id, pos_new, rot)
                init_positions[idx] = pos_new
                #print(cal_distance(objects_pair))

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        init_rotations = np.array(init_rotations)

        is_feasible = self.feasibility_check(selected_objects, init_positions, init_rotations) if not random else self.stable_check(selected_objects)
        data_objects_list = {id: self.objects_list[id] for id in selected_objects}
        is_on_table = check_on_table(data_objects_list)
        is_leaning_obj = self.leaning_check(selected_objects)
        is_feasible = is_feasible and is_on_table and not is_leaning_obj

        return is_feasible, data_objects_list

    def load_template(self, scene_id, template_file): 
        # 기존 방법이랑 다르게 매번 새로운 object들을 spawn해서 수집.
        # template load후 messup을 한 다음에는, scene clear 해줘야함.        
        with open(template_file, 'r') as f:
            templates = json.load(f)
        self.pre_selected_objects = []
        template_id_to_sim_id = {}
        data_objects_list = {}
        template_id_to_obj = {}
        spawn_list = []
        for action in templates['action_order']:
            action_type, obj_id, pos, rot = action
            if action_type == 0: # spawn
                obj_cat = templates['objects'][str(obj_id)]
                if 'large' in obj_cat or 'small' in obj_cat:
                    obj_size = obj_cat.split('_')[0]
                    obj_cat = ''.join(obj_cat.split('_')[1:])
                else:
                    obj_size = 'medium'
                obj_name = np.random.choice(object_cat_to_name[obj_cat])               
                spawn_list.append((obj_name, obj_size))
                obj_name = obj_name.split('/')[-1]
                template_id_to_obj[obj_id] = (obj_name, obj_size)
        
        self.spawn_objects(spawn_list)
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        for action in templates['action_order']:
            action_type, obj_id, pos, rot = action
            if action_type == 0: # spawn
                obj = template_id_to_obj[obj_id]
                key_list = list(self.objects_list.keys())
                value_list = list(self.objects_list.values())
                sim_obj_id = key_list[value_list.index(obj)]
                template_id_to_sim_id[obj_id] = sim_obj_id
                data_objects_list[sim_obj_id] = self.objects_list[sim_obj_id]
                p.resetBasePositionAndOrientation(sim_obj_id, pos, rot)
                
            elif action_type == 1:
                sim_obj_id = template_id_to_sim_id[obj_id]
                p.resetBasePositionAndOrientation(sim_obj_id, pos, rot)                
                
            elif action_type == 2:
                for _ in range(100):
                    p.stepSimulation()
        self.pre_selected_objects = list(template_id_to_sim_id.values())
        # TODO : check feasibility if needed

        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)

        self.save_data(scene_id, data_objects_list)
        
    # need to change. spherical objects are not considered.
    def feasibility_check(self, selected_objects, init_positions, init_rotations):
        init_feasible = False 
        j = 0
        while j<2000 and not init_feasible:
            p.stepSimulation()
            if j%10==0:
                current_poses = []
                current_rotations = []
                for obj_id in selected_objects:
                    pos, rot = p.getBasePositionAndOrientation(obj_id)
                    current_poses.append(pos)
                    current_rotations.append(rot)

                current_poses = np.array(current_poses)
                current_rotations = np.array(current_rotations)
                pos_diff = np.linalg.norm(current_poses[:, :2] - init_positions[:, :2], axis=1)
                rot_diff = np.linalg.norm(current_rotations - init_rotations, axis=1)
                if (pos_diff > self.threshold['pose']).any() or (rot_diff > self.threshold['rotation']).any():
                    break
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    init_feasible = True
            j += 1
        return init_feasible
    
    def stable_check(self, selected_objects):
        init_stable = False
        j = 0
        while j<1000 and not init_stable:
            p.stepSimulation()
            if j % 10 == 0:
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    init_stable = True
            j+=1
            
        return init_stable

    def leaning_check(self, selected_objects):
        contacts = get_contact_objects()
        is_leaning_obj = False
        leaning_obj_list = []
        for obj in selected_objects:
            contact_table = False
            contact_obj = False
            for s in contacts:
                if obj in s and 1 in s:
                    contact_table = True
                if obj in s and 1 not in s:
                    contact_obj = True
            if contact_table and contact_obj:
                leaning_obj_list.append(obj)
        for obj1 in leaning_obj_list:
            for obj2 in leaning_obj_list:
                if (obj1, obj2) in contacts or (obj2, obj1) in contacts:
                    is_leaning_obj = True
                    break
        return is_leaning_obj
            
                
    # TODO : change this function to move pickable objects to a random place
    def random_messup_objects(self, scene_id, random_rot = True):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        selected_objects = copy.deepcopy(self.pre_selected_objects)
        data_objects_list = {id: self.objects_list[id] for id in selected_objects}

        #select pickable object
        move_obj_id = np.random.choice(self.pickable_objects)
        
        # save current poses & rots #
        pos_saved, rot_saved = {}, {}
        for obj_id in selected_objects:
            pos, rot = p.getBasePositionAndOrientation(obj_id)
            pos_saved[obj_id] = pos
            rot_saved[obj_id] = rot

        # set poses & rots #
        
        place_feasible = False
        count_scene_repose = 0
        while not place_feasible:
            # get the pose of the objects
            pos, rot = p.getBasePositionAndOrientation(move_obj_id)
            collisions_before = get_contact_objects()

            if random_rot:
                pos_new, rot = get_random_pos_orn(rot)
            else :
                pos = random_pos_on_table()
                rot = self.base_rot[move_obj_id]
            if self.opt.mess_grid:
                pos = get_random_pos_from_grid()
            p.resetBasePositionAndOrientation(move_obj_id, pos_new, rot)
            collisions_after = set()
            for _ in range(200):
                p.stepSimulation()
                collisions_after = collisions_after.union(get_contact_objects())

            collisions_new = collisions_after - collisions_before
            if len(collisions_new) > 0:
                place_feasible = False

                # reset non-target objects
                obj_to_reset = set()
                for collision in collisions_new:
                    obj1, obj2 = collision
                    obj_to_reset.add(obj1)
                    obj_to_reset.add(obj2)
                obj_to_reset = obj_to_reset - set([move_obj_id]) - set([1]) # table id
                for reset_obj_id in obj_to_reset:
                    if reset_obj_id in selected_objects:
                        p.resetBasePositionAndOrientation(reset_obj_id, pos_saved[reset_obj_id], rot_saved[reset_obj_id])
            else:
                place_feasible = True
                pos, rot = p.getBasePositionAndOrientation(move_obj_id)
                pos_saved[move_obj_id] = pos
                rot_saved[move_obj_id] = rot
            count_scene_repose += 1
            is_on_table = check_on_table(data_objects_list)
            is_leaning_obj = self.leaning_check(selected_objects)
            place_feasible = place_feasible and is_on_table and not is_leaning_obj

            if count_scene_repose > 10:
                break

        if not place_feasible:
            return False

        for j in range(2000):
            p.stepSimulation()
            if j%10==0:
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    break
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        
        self.save_data(scene_id, data_objects_list)
        return True

    def render_and_save_scene(self,out_folder, camera):
        if camera == 'top':
            nv.set_camera_entity(self.camera_top)
        elif camera == 'front_top':
            nv.set_camera_entity(self.camera_front_top)
        nv.render_to_file(
            width=int(self.opt.width), height=int(self.opt.height), 
            samples_per_pixel=int(self.opt.spp),
            file_path=f"{out_folder}/rgb_{camera}.png"
        )
        d = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=5, bounce=0, options='depth',
        )
        depth = np.array(d).reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        depth = np.flip(depth, axis = 0)
        depth[np.isinf(depth)] = 3
        depth[depth < 0] = 3
        np.save(f"{out_folder}/depth_{camera}.npy", depth)
        
        entity_id = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=1, bounce=0, options='entity_id',
        )
        entity = np.array(entity_id)
        entity = entity.reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        entity = entity - 2
        entity = np.flip(entity, axis = 0)
        entity[np.isinf(entity)] = -1
        entity[entity>self.opt.nb_objects + 50] = -1
        np.save(f"{out_folder}/seg_{camera}.npy", entity)
        return

    def save_data(self, scene_idx, objects_list):
        """Save data to a file."""
        obj_info = {}
        out_folder = f"{self.opt.out_folder}/{self.opt.dataset}/{scene_idx['scene']}/template_{str(scene_idx['template_id']).zfill(5)}/traj_{str(scene_idx['trajectory']).zfill(5)}/{str(scene_idx['frame']).zfill(3)}"
        # out_folder = f"{self.opt.out_folder}/{self.opt.save_scene_name}/template_{str(scene_idx['template']).zfill(5)}/traj_{str(scene_idx['trajectory']).zfill(5)}/{str(scene_idx['frame']).zfill(3)}"
        if os.path.isdir(out_folder):
            print(f'folder {out_folder}/ exists')
        else:
            os.makedirs(out_folder)
            print(f'created folder {out_folder}/')
        self.render_and_save_scene(out_folder,'top')
        self.render_and_save_scene(out_folder,'front_top')
        
        obj_semantic_label = {}
        obj_aabb = {}
        obj_adjacency = {}
        obj_state = {}
        n_obj = {}
        sizes = {}
        new_objects_list = {}
        for id, (obj_name, size) in objects_list.items():
            id = int(id)
            new_objects_list[id] = obj_name
            n = n_obj.get(obj_name_to_semantic_label[obj_name], 0)
            n_obj[obj_name_to_semantic_label[obj_name]] = n + 1
            obj_semantic_label[id] = obj_name_to_semantic_label[obj_name] + '_' + str(n)
            sizes[id] = size
            obj_aabb[id] = p.getAABB(id)
            obj_state[id] = p.getBasePositionAndOrientation(id)
            overlapping_objs = p.getOverlappingObjects(obj_aabb[id][0], obj_aabb[id][1])
            obj_adjacency[id] = []
            for obj in overlapping_objs:
                obj_adjacency[id].append(obj[0])
        
        obj_info['objects'] = new_objects_list
        obj_info['semantic_label'] = obj_semantic_label
        obj_info['obj_aabb'] = obj_aabb
        obj_info['sizes'] = sizes
        obj_info['distances'] = cal_distance(new_objects_list)
        obj_info['state'] = obj_state
        sg = generate_sg(obj_info)
        obj_info['pickable_objects'] = pickable_objects_list(new_objects_list, sg)
        obj_info['scene_graph'] = sg
        self.pickable_objects = obj_info['pickable_objects']
        with open(out_folder+"/obj_info.json", "w") as f:
            json.dump(obj_info, f)
        
        return




if __name__=='__main__':
    opt = lambda : None
    opt.nb_objects = 8 #20 #20
    opt.nb_objects = 25 #20
    opt.inscene_objects = 5 #7
    opt.scene_type = 'random' # 'random' or 'line'
    opt.spp = 32 #64 
    opt.width = 480
    opt.height = 360
    opt.noise = False
    opt.mess_grid = True
    opt.nb_frames = 5 #7
    # opt.out_folder = '/ssd/ur5_tidying_data/line-shape/'
    opt.out_folder = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/dataset'
    opt.nb_randomset = 20
    opt.num_traj = 100
    opt.dataset = 'train' #'train' or 'test'
    opt.objectset = 'all' # 'pybullet' #'pybullet'/'ycb'/ 'housecat'/ 'all'
    # opt.pybullet_object_path = '/ssd/pybullet-URDF-models/urdf_models/models'
    # opt.ycb_object_path = '/ssd/YCB_dataset'
    # opt.ig_object_path = '/ssd/ig_dataset/objects'
    # opt.housecat_object_path = '/ssd/housecat6d/obj_models_small_size_final'
    opt.pybullet_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/pybullet-URDF-models/urdf_models/models'
    opt.ycb_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/YCB_dataset'
    opt.housecat_object_path = '/home/wooseoko/workspace/hogun/pybullet_scene_gen/TabletopTidyingUp/housecat6d/obj_models_small_size_final'

    if os.path.isdir(opt.out_folder):
        print(f'folder {opt.out_folder}/ exists')
    else:
        os.makedirs(opt.out_folder)
        print(f'created folder {opt.out_folder}/')
        
    ### use template ###
    # template_folder = './templates'
    # template_files = os.listdir(template_folder)
    # template_files = [f for f in template_files if f.lower().endswith('.json')]
    # collect_scenes = ['env1-s1'] #... train/test 나눠서 수집. test에 들어가는거 : unseen template, unseen obj + seen template. 이거 두개도 나눠서 수집해야할듯,
    # ts = TabletopScenes(opt)
    # for template_file in template_files:
    #     if template_file.split('_')[0] in collect_scenes:
            
    #         for traj_id in range(100): # collect trajectory num
    #             scene = template_file.split('_')[0]
    #             template_id = template_file.split('_')[-1].split('.')[0]
    #             scene_id = {'scene': scene, 'template_id': template_id,'trajectory': traj_id, 'frame': 0}
                
    #             ts.set_floor(texture_id=-1)
    #             print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
    #             success_placement = ts.load_template(scene_id, os.path.join(template_folder, template_file))
    #             scene_id['frame'] += 1
                
    #             # 2. Move each object to a random place #
    #             while scene_id['frame'] < int(opt.nb_frames): #
    #                 print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
    #                 success_placement = ts.random_messup_objects(scene_id)
    #                 if not success_placement:
    #                     continue
    #                 scene_id['frame'] += 1
                    
    #             ts.clear()
                
    #     ts.clear()
    # ts.close()

    spawn_objects_list = object_name_list #['stapler_2', 'two_color_hammer', 'scissors', 'extra_large_clamp', 'phillips_screwdriver', 'stapler_1', 'conditioner', 'book_1', 'book_2', 'book_3', 'book_4', 'book_5', 'book_6', 'power_drill', 'plastic_pear', 'cracker_box', 'blue_plate', 'blue_cup', 'cleanser', 'bowl', 'plastic_lemon', 'mug', 'square_plate_4', 'sugar_box', 'plastic_strawberry', 'medium_clamp', 'plastic_peach', 'knife', 'square_plate_2', 'fork', 'plate', 'green_cup', 'green_bowl', 'orange_cup', 'large_clamp', 'spoon', 'pink_tea_box', 'pudding_box', 'plastic_orange', 'plastic_apple', 'doraemon_plate', 'lipton_tea', 'yellow_bowl', 'grey_plate', 'gelatin_box', 'blue_tea_box', 'flat_screwdriver', 'mini_claw_hammer_1', 'shampoo', 'glue_1', 'glue_2', 'small_clamp', 'square_plate_3', 'doraemon_bowl', 'square_plate_1', 'round_plate_1', 'round_plate_3', 'round_plate_2', 'round_plate_4', 'plastic_banana', 'yellow_cup']
    #spawn_objects_list = ['bottle-85_alcool', 'bottle-nivea', 'can-fanta', 'can-redbull', 'cup-grey_handle', 'cup-new_york', 'cup-stanford', 'remote-black', 'remote-toy', 'teapot-blue_floral']
    scenes = ['random_4', 'random_5', 'random_6', 'random_7'] #train 
    # scenes = ['random_3', 'random_4', 'random_5'] #test
    ts = TabletopScenes(opt)
    for scene in scenes:
        for n_set in range(opt.nb_randomset): 
            n_obj = int(scene.split('_')[-1])
            opt.inscene_objects = n_obj
            spawn_list = np.random.choice(spawn_objects_list, opt.nb_objects, replace=False)
            spawn_list = [(f, 'medium') for f in spawn_list]
            ts.spawn_objects(spawn_list) # add random select or template load
            for i in range(opt.num_traj):
                traj_id = i # + num_exist_trajs
                #############################
                
                scene_id = {'scene': scene, 'template_id': n_set,'trajectory': traj_id, 'frame': 0}           
                success_placement = False
                while not success_placement:
                    ts.set_floor(texture_id=-1)
                    print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
                    success_placement = ts.arrange_objects(scene_id, random=True)
                    if not success_placement:
                        continue
                scene_id['frame'] += 1
                
                # 2. Move each object to a random place #
                while scene_id['frame'] < int(opt.nb_frames): #
                    print(f'rendering scene {str(scene_id["scene"])}-{str(scene_id["template_id"])}-{str(scene_id["trajectory"])}-{str(scene_id["frame"])}', end='\r')
                    success_placement = ts.random_messup_objects(scene_id)
                    if not success_placement:
                        continue
                    scene_id['frame'] += 1
                    
            ts.clear()
        ts.close()
