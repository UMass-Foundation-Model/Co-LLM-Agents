from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.floorplan import Floorplan
from tdw.tdw_utils import TDWUtils
from tdw.librarian import SceneLibrarian, ModelLibrarian
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.scene_data.scene_bounds import SceneBounds
from utils import *
import random
import json
import numpy as np

import sys
import os

def check_maximum_count(obj_name, current_count, container_limit = 6, round = 0):
    hard_bar = 4 if round == 0 else 6
    if obj_name in object_place['food']['target']: 
        if current_count["count_target_food"] >= 10: return False
    if obj_name in object_place['food']['container']:
        if current_count["count_container_food"] >= container_limit: return False
    if obj_name in object_place['stuff']['target']:
        if current_count["count_target_stuff"] >= 10: return False
    if obj_name in object_place['stuff']['container']:
        if current_count["count_container_stuff"] >= container_limit: return False
    if obj_name in object_place['food']['target_fruit']:
        if current_count["count_target_food_fruit"] >= hard_bar: return False
    if obj_name in object_place['food']['target_bread']:
        if current_count["count_target_food_bread"] >= hard_bar: return False
    if obj_name in object_place['stuff']['target_office']:
        if current_count["count_target_stuff_office"] >= hard_bar: return False
    if obj_name in object_place['stuff']['target_common']:
        if current_count["count_target_stuff_common"] >= hard_bar: return False
    return True

def update_maximum_count(obj_name, current_count):
    if obj_name in object_place['food']['target']: current_count["count_target_food"] += 1
    if obj_name in object_place['food']['container']: current_count["count_container_food"] += 1
    if obj_name in object_place['stuff']['target']: current_count["count_target_stuff"] += 1
    if obj_name in object_place['stuff']['container']: current_count["count_container_stuff"] += 1
    if obj_name in object_place['food']['target_fruit']: current_count["count_target_food_fruit"] += 1
    if obj_name in object_place['food']['target_bread']: current_count["count_target_food_bread"] += 1
    if obj_name in object_place['stuff']['target_office']: current_count["count_target_stuff_office"] += 1
    if obj_name in object_place['stuff']['target_common']: current_count["count_target_stuff_common"] += 1

def check_map(x, z, occ_map, positions, threshold = 0.65):
    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            if occ_map[i, j] == 1 and (positions[i, j][0] - positions[x, z][0]) ** 2 + (positions[i, j][1] - positions[x, z][1]) ** 2 < threshold ** 2:
                return False
    return True

#scene settings
settings = sys.argv[1:]
FLOORPLAN_SCENE_NAME = settings[0]
FLOORPLAN_LAYOUT = int(settings[1])
scene_id = int(settings[2])
dataset_prefix = settings[3]
print("FLOORPLAN_SCENE_NAME:", FLOORPLAN_SCENE_NAME)
print("FLOORPLAN_LAYOUT:", FLOORPLAN_LAYOUT)
print("scene_id:", scene_id)

container_limit, container_room_limit = 0, 0
if scene_id == 0: container_limit, container_room_limit = 5, 2
if scene_id == 1: container_limit, container_room_limit = 2, 1
if scene_id == 2: container_limit, container_room_limit = 4, 1

#start simulator
c = Controller(port = 1077)

occ = OccupancyMap()
occ.generate(cell_size=0.25, once=False)
om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
c.add_ons.extend([occ, om])

floorplan = Floorplan()

floorplan.init_scene(scene=FLOORPLAN_SCENE_NAME, layout=FLOORPLAN_LAYOUT)

with open('./dataset/list.json', 'r') as f:
    object_place = json.loads(f.read())

with open('./dataset/name_map.json', 'r') as f:
    name_map = json.loads(f.read())

with open('./dataset/object_scale.json', 'r') as f:
    object_scale = json.loads(f.read())

os.makedirs("./dataset/", exist_ok=True)
os.makedirs(f"./dataset/{dataset_prefix}/", exist_ok=True)

random_object_list_on_floor = object_place['floor_objects']

commands_init_scene = [{"$type": "set_screen_size", "width": 1920, "height": 1080}] # Set screen size
commands_init_scene.extend(floorplan.commands)
commands_init_scene.append({"$type": "set_floorplan_roof", "show": False}) # Hide roof

response = c.communicate(commands_init_scene)

resp = c.communicate([{"$type": "send_scene_regions"}])
scene_bounds = SceneBounds(resp=resp)

commands = []
object_room_list = []
camera = ThirdPersonCamera(position={"x": 0, "y": 30, "z": 0},
                           look_at={"x": 0, "y": 0, "z": 0},
                           avatar_id="a")

screenshot_path = f"./dataset/{dataset_prefix}/screenshots/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}_{scene_id}"
if not os.path.exists(screenshot_path):
    os.makedirs(screenshot_path)
print(f"Images will be saved to: {screenshot_path}")

capture = ImageCapture(avatar_ids=["a"], path=screenshot_path, pass_masks=["_img"])
commands.extend(camera.get_initialization_commands()) # Init camera
c.add_ons.extend([capture]) # Capture images

def get_scale(obj_name, object_scale):
    if obj_name in object_scale:
        return object_scale[obj_name]
    else:
        print("WARNING: No scale for object: ", obj_name)
        return 1
    
def object_choice(object_list):
    assert len(object_list) % 3 == 0, "Invalid object list"
    if len(object_list) == 3:
        return object_list[random.randint(0, 2)]
    elif len(object_list) == 6:
        if random.random() < 0.5:
            return object_list[random.randint(0, 2)]
        else:
            return object_list[random.randint(3, 5)]
    elif len(object_list) == 12: # Objects in front have greater priority
        if random.random() < 0.5:
            return object_list[random.randint(0, 2)]
        else:
            return object_list[random.randint(3, 5)]


count = {}
current_count = {
    "count_target_food": 0,
    "count_container_food": 0,
    "count_target_stuff": 0,
    "count_container_stuff": 0,
    "count_target_food_fruit": 0,
    "count_target_food_bread": 0,
    "count_target_stuff_office": 0,
    "count_target_stuff_common": 0
}

place_list = []
for object_id in om.objects_static:
    place_list.append(object_id)
np.random.shuffle(place_list)

for object_id in place_list:
    id = belongs_to_which_room(om.bounds[object_id].top[0], om.bounds[object_id].top[2], scene_bounds)
    if id == -1: continue
    func_name = get_room_functional_by_id(FLOORPLAN_SCENE_NAME, FLOORPLAN_LAYOUT, id)
    if om.objects_static[object_id].category in ['table', 'sofa', 'chair']: # place objects on specific objects
        if om.objects_static[object_id].category in ['table']:
            num = np.random.randint(1, 3) # more than 3 objects on a table is too crowded
        else:
            num = np.random.randint(0, 2)
        for i in range(num):
            if len(object_place[om.objects_static[object_id].category][func_name]) == 0: continue
            obj = c.get_unique_id()
            p = shift(om.bounds[object_id], i)
            obj_name = object_choice(object_place[om.objects_static[object_id].category][func_name])
            if not check_maximum_count(obj_name, current_count, container_limit, round = 0):
                continue
            update_maximum_count(obj_name, current_count)
            if obj_name not in count: count[obj_name] = 0
            count[obj_name] += 1
            object_room_list.append([obj_name, func_name])
            commands.extend(c.get_add_physics_object(obj_name,
                                    object_id=obj,
                                    position=TDWUtils.array_to_vector3(p),
                                    scale_factor = {
                                        "x": get_scale(obj_name, object_scale),
                                        "y": get_scale(obj_name, object_scale),
                                        "z": get_scale(obj_name, object_scale),
                                    }))

object_probability = 1
curr_occ_map = occ.occupancy_map.copy()
food_container_room_count, stuff_container_room_count = 0, 0
has_container = {}
for _ in range(occ.occupancy_map.shape[0] * occ.occupancy_map.shape[1] * 3):
        x_index = np.random.randint(0, occ.occupancy_map.shape[0])
        z_index = np.random.randint(0, occ.occupancy_map.shape[1])
        if check_map(x_index, z_index, curr_occ_map, occ.positions, 0.5): # Unoccupied
            if np.random.random() < object_probability:
                x = float(occ.positions[x_index, z_index][0])
                z = float(occ.positions[x_index, z_index][1])
                room_id = belongs_to_which_room(x, z, scene_bounds)
                if room_id == -1: continue
                func_name = get_room_functional_by_id(FLOORPLAN_SCENE_NAME, FLOORPLAN_LAYOUT, room_id)
                if room_id < 0 or func_name in ['Bedroom', 'Porch']: # cannot place objects in bedroom or porch
                    continue
                else:
                    obj_name = np.random.choice(random_object_list_on_floor)
                if not check_maximum_count(obj_name, current_count, container_limit, round = 1):
                    continue
                if obj_name in object_place['food']['target']:
                    if obj_name not in object_place['ground'][func_name]: continue
                if obj_name in object_place['food']['container']:
                    if not(room_id not in has_container or has_container[room_id] == 'food'): continue
                    if 'Office' == func_name or not check_map(x_index, z_index, curr_occ_map, occ.positions): continue
                    if room_id not in has_container:
                        if food_container_room_count >= container_room_limit:
                            continue 
                        has_container[room_id] = 'food'
                        food_container_room_count += 1
                if obj_name in object_place['stuff']['target']:
                    if obj_name not in object_place['ground'][func_name]: continue
                if obj_name in object_place['stuff']['container']:
                    if not(room_id not in has_container or has_container[room_id] == 'stuff'): continue
                    if 'Kitchen' == func_name or not check_map(x_index, z_index, curr_occ_map, occ.positions): continue
                    if room_id not in has_container:
                        if stuff_container_room_count >= container_room_limit:
                            continue 
                        has_container[room_id] = 'stuff'
                        stuff_container_room_count += 1
                update_maximum_count(obj_name, current_count)
                object_room_list.append([obj_name, func_name])
                if obj_name not in count: count[obj_name] = 0
                count[obj_name] += 1
                commands.extend(c.get_add_physics_object(obj_name,
                                                 object_id=c.get_unique_id(),
                                                 position={"x": x, "y": 0.1, "z": z},
                                                 rotation={"x": 0, "y": np.random.uniform(0, 360), "z": 0},
                                                 scale_factor = {
                                                 "x": get_scale(obj_name, object_scale),
                                                 "y": get_scale(obj_name, object_scale),
                                                 "z": get_scale(obj_name, object_scale),
                                                }))
                curr_occ_map[x_index][z_index] = 1
                print(x_index, z_index, x, z, room_id, obj_name, occ.positions[x_index, z_index])
                
commands.append({"$type": "step_physics", "frames": 100})

print(count)
print("Total number of objects: ", sum(count.values()))
print("Total number of target food: ", current_count['count_target_food'])
print("Total number of container food: ", current_count['count_container_food'])
print("Total number of target stuff: ", current_count['count_target_stuff'])
print("Total number of container stuff: ", current_count['count_container_stuff'])

count_goal_container = {
    'food': {
        'target': current_count['count_target_food'],
        'container': current_count['count_container_food'],
    },
    'stuff': {
        'target': current_count['count_target_stuff'],
        'container': current_count['count_container_stuff'],
    }
}

metadata = {
    'floorplan_scene_name': FLOORPLAN_SCENE_NAME,
    'floorplan_layout': FLOORPLAN_LAYOUT,
    'scene_id': scene_id,
    'object_room_list': object_room_list,
    **count_goal_container,
}

# Save commands and metadata to json


with open(f"./dataset/{dataset_prefix}/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}_{scene_id}.json", "w") as f:
    f.write(json.dumps(commands, indent=4))
with open(f"./dataset/{dataset_prefix}/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}_{scene_id}_metadata.json", "w") as f:
    f.write(json.dumps(metadata, indent=4))

response = c.communicate(commands)

c.communicate({"$type": "terminate"})
