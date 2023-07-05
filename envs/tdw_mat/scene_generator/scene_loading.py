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
from random import choice
import json
import numpy as np

c = Controller(port = 1075)
floorplan = Floorplan()
FLOORPLAN_SCENE_NAME = "2a"
FLOORPLAN_LAYOUT = 0
floorplan.init_scene(scene=FLOORPLAN_SCENE_NAME, layout=FLOORPLAN_LAYOUT)
commands_init_scene = [{"$type": "set_screen_size", "width": 1920, "height": 1080}] # Set screen size
commands_init_scene.extend(floorplan.commands)
commands_init_scene.append({"$type": "set_floorplan_roof", "show": False}) # Hide roof

response = c.communicate(commands_init_scene)
resp = c.communicate([{"$type": "send_scene_regions"}])
scene_bounds = SceneBounds(resp=resp)

with open(f'./dataset/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}.json', 'r') as f:
    all_commands = json.loads(f.read())

c.communicate(all_commands)
