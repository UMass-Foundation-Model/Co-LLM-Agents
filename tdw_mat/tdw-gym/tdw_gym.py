import string
from typing import Optional

import gym
from gym.core import Env
import numpy as np
import os
import time
import copy

from tdw.replicant.arm import Arm
from tdw.tdw_utils import TDWUtils

from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from collections import Counter
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.object_manager import ObjectManager
from PIL import Image

import json
import pickle
from functools import partial
import signal
from tenacity import retry, wait_fixed, retry_if_exception_type

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded the timeout limit")

@retry(wait=wait_fixed(5), retry=retry_if_exception_type(TimeoutException))  # wait 5 seconds between retries
def might_fail_launch(launch, port = None):
    if port is not None:
        print("kill failure launch ...", f"ps ux | grep TDW.x86_64\ -port\ {port} | awk {{'print $2'}} | xargs kill")
        os.system(f"ps ux | grep TDW.x86_64\ -port\ {port} | awk {{'print $2'}} | xargs kill")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)
    try:
        print("Trying to launch tdw ...")
        return launch()
    finally:
        signal.alarm(0)  

class TDW(Env):
    def __init__(self, port = 1071, number_of_agents = 1, demo=False, rank=0, num_scenes = 0, train=False, \
                        screen_size = 512, exp = False, launch_build=True, gt_occupancy = False, gt_mask = True, enable_collision_detection = False, save_dir = 'results', max_frames = 3000, data_prefix = 'dataset/nips_dataset/'):
        self.messages = None
        self.data_prefix = data_prefix
        self.replicant_colors = None
        self.replicant_ids = None
        self.names_mapping = None
        self.rooms_name = None
        self.action_buffer = None
        self.scene_bounds = None
        self.goal_description = None
        self.object_manager = None
        self.occupancy_map = None
        self.gt_mask = gt_mask
        self.satisfied = None
        self.count = 0
        self.reach_threshold = 2
        self.number_of_agents = number_of_agents
        self.seed = None
        self.num_step = 0
        self.reward = 0
        self.done = False
        self.scene_info = None
        self.exp = exp
        self.success = False
        self.num_frames = 0
        self.data_id = rank     
        self.train = train
        self.port = port
        self.gt_occupancy = gt_occupancy
        self.screen_size = screen_size
        self.launch_build = launch_build
        self.enable_collision_detection = enable_collision_detection
        self.controller = None
        self.message_per_frame = 500
        rgb_space = gym.spaces.Box(0, 256,
                                 (3,
                                  self.screen_size,
                                  self.screen_size), dtype=np.int32)
        seg_space = gym.spaces.Box(0, 256, \
                                (self.screen_size, \
                                self.screen_size, \
                                3), dtype=np.int32)
        depth_space = gym.spaces.Box(0, 256, \
                                (self.screen_size, \
                                self.screen_size), dtype=np.int32)
        object_space = gym.spaces.Dict({
            'id': gym.spaces.Discrete(30),
            'type': gym.spaces.Discrete(4),
            'seg_color': gym.spaces.Box(0, 255, (3, ), dtype=np.int32),
            'name': gym.spaces.Text(max_length=100, charset=string.printable)
        })

        self.action_space_single = gym.spaces.Dict({
            'type': gym.spaces.Discrete(7), # please refer to line 192-210
            'object': gym.spaces.Discrete(30),
            'arm': gym.spaces.Discrete(2),
            'message': gym.spaces.Text(max_length=1000, charset=string.printable)
        })
        
        self.hand_object_space = gym.spaces.Dict({
            'id': gym.spaces.Discrete(30),
            'type': gym.spaces.Discrete(4),
            'name': gym.spaces.Text(max_length=100, charset=string.printable),
            'contained': gym.spaces.Tuple(gym.spaces.Discrete(30) for _ in range(3)),
            'contained_name': gym.spaces.Tuple(gym.spaces.Text(max_length=100, charset=string.printable) for _ in range(3))
        })
        
        self.observation_space_single = gym.spaces.Dict({
            'rgb': rgb_space,
            'seg_mask': seg_space,
            'depth': depth_space,
            'agent': gym.spaces.Box(-30, 30, (6, ), dtype=np.float32),
            'held_objects': gym.spaces.Tuple((self.hand_object_space, self.hand_object_space)),
            'oppo_held_objects': gym.spaces.Tuple((self.hand_object_space, self.hand_object_space)),
            'visible_objects': gym.spaces.Tuple(object_space for _ in range(50)),
            'status': gym.spaces.Discrete(4),
            'valid': gym.spaces.Discrete(2),
            'FOV': gym.spaces.Box(0, 120, (1,), dtype=np.float32),
            'camera_matrix': gym.spaces.Box(-30, 30, (4, 4), dtype=np.float32),
            'messages': gym.spaces.Tuple(gym.spaces.Text(max_length=1000, charset=string.printable) for _ in range(2)),
            'current_frames': gym.spaces.Discrete(30),
        })

        self.observation_space = gym.spaces.Dict({
            str(i): self.observation_space_single for i in range(self.number_of_agents)
        })

        self.action_space = gym.spaces.Dict({
            str(i): self.observation_space_single for i in range(self.number_of_agents)
        })
        self.max_frame = max_frames
        self.f = open(f'action{port}.log', 'w')
        self.action_list = []
                    
        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.target_object_ids = []
        self.container_ids = []
        self.goal_position_id = None # The place to put the object
        self.fov = 0
        self.save_dir = save_dir
    
    def obs_filter(self, obs):
        if self.gt_mask:
            return obs
        else:
            new_obs = copy.deepcopy(obs)
            for agent in obs:
                new_obs[agent]['seg_mask'] = np.zeros_like(new_obs[agent]['seg_mask'])
                new_obs[agent]['visible_objects'] = []
                while len(new_obs[agent]['visible_objects']) < 50:
                    new_obs[agent]['visible_objects'].append({
                        'id': None,
                        'type': None,
                        'seg_color': None,
                        'name': None,
                    })
            return new_obs

    def get_object_type(self, id):
        if id in self.target_object_ids:
            return 0 # target object
        if id in self.container_ids:
            return 1 # container
        if self.object_categories[id] == 'bed':
            return 2 # goal position
        # 3: agent
        # 4: obstacle
        return 5 # unrelated object

    def get_with_character_mask(self, agent_id, character_object_ids):
        color_set = [self.segmentation_colors[id] for id in character_object_ids if id in self.segmentation_colors] + [self.replicant_colors[id] for id in character_object_ids if id in self.replicant_colors]
        curr_with_seg = np.zeros_like(self.obs[str(agent_id)]['seg_mask'])
        curr_seg_flag = np.zeros((self.screen_size, self.screen_size), dtype = bool)
        for i in range(len(color_set)):
            color_pos = (self.obs[str(agent_id)]['seg_mask'] == np.array(color_set[i])).all(axis=2)
            curr_seg_flag = np.logical_or(curr_seg_flag, color_pos)
            curr_with_seg[color_pos] = color_set[i]
        return curr_with_seg, curr_seg_flag
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        reset the environment
        input:
            data_id: reset based on the data_id
        """
        # Changes it to always, since in each step, we need to get the image
        if self.controller is not None:
            self.controller.communicate({"$type": "terminate"})
            self.controller.socket.close()
        # download_asset_bundles()
        self.controller = might_fail_launch(partial(TransportChallenge, port=self.port, check_version=True, launch_build=self.launch_build, screen_width=self.screen_size,screen_height=self.screen_size, image_frequency= ImageFrequency.always, png=True, image_passes=None, enable_collision_detection = self.enable_collision_detection, logger_dir = output_dir), port = self.port)
        print("Controller connected")
        self.success = False
        self.messages = [None for _ in range(self.number_of_agents)]
        self.reward = 0
        scene_info = options
        print(scene_info)
        self.satisfied = {}
        if output_dir is not None: self.save_dir = output_dir
        if scene_info is not None:
            scene = scene_info['scene']
            layout = scene_info['layout']
            if 'task' in scene_info:
                task = scene_info['task']
            else:
                task = None
        else: raise ValueError("No scene info assigned!")
        super().reset(seed=seed)
        self.seed = np.random.RandomState(seed)
        self.scene_info = scene_info
        
        # Now the scene is fixed, so num_containers and num_target_objects are not used anymore in new settings
        self.controller.start_floorplan_trial(scene=scene, layout=layout, replicants=self.number_of_agents, num_containers=4, num_target_objects=10,
                                   random_seed=seed, task = task, data_prefix = self.data_prefix)

        # Add a gt occupancy map. In the standard setting, we don't need this
        if self.gt_occupancy:
            self.occupancy_map = OccupancyMap()
            self.controller.add_ons.append(self.occupancy_map)
            self.occupancy_map.generate(cell_size=0.125, once = False)
        self.controller.communicate({"$type": "set_floorplan_roof",
                          "show": False})

        # Bright case   
        self.controller.communicate({"$type": "add_hdri_skybox", "name": "sky_white", "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white", "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90, "sun_initial_angle": 0, "sun_intensity": 1.25})
            
        # Set the field of view of the agent.
        for replicant_id in self.controller.replicants:
            self.controller.communicate({"$type": "set_field_of_view",
                            "avatar_id" : str(replicant_id), "field_of_view" : 90})
        self.fov = 90
        
        # Add a object manager for object position
        self.object_manager = ObjectManager()
        self.controller.add_ons.append(self.object_manager)

        data = self.controller.communicate({"$type": "send_segmentation_colors",
                          "show": False,
                          "frequency": "once"})
        
        # Show the occupancy map. In the standard setting, we don't need this
        if self.gt_occupancy:            
            self.occupancy_map.show()
            print(self.occupancy_map.occupancy_map)
            h, w = self.occupancy_map.occupancy_map.shape
            print(self.occupancy_map.occupancy_map.shape)

        # Make name easier to read
        names_mapping_path = f'./dataset/name_map.json'
        with open(names_mapping_path, 'r') as f: self.names_mapping = json.load(f)

        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.target_object_ids = self.controller.state.target_object_ids
        self.container_ids = self.controller.state.container_ids
        self.replicant_ids = [self.controller.replicants[i].static.replicant_id for i in range(self.number_of_agents)]
        
        for i in range(len(data) - 1):
            r_id = OutputData.get_data_type_id(data[i])
            if r_id == "segm":
                segm = SegmentationColors(data[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    self.segmentation_colors[object_id] = segm.get_object_color(j)
                    self.object_names[object_id] = segm.get_object_name(j).lower()
                    if self.object_names[object_id] in self.names_mapping:
                        self.object_names[object_id] = self.names_mapping[self.object_names[object_id]]
                    self.object_categories[object_id] = segm.get_object_category(j)
                    if self.object_categories[object_id] == 'bed':
                        self.goal_position_id = object_id
        
        self.replicant_colors ={i: self.controller.replicants[i].static.segmentation_color for i in range(self.number_of_agents)}

        self.containment_all = {}
        
        # check colors are different:
        for x in self.segmentation_colors.keys():
            for y in self.segmentation_colors.keys():
                if x != y: assert (self.segmentation_colors[x] != self.segmentation_colors[y]).any()

        self.num_step = 0
        self.num_frames = 0
        self.goal_description = {}
        for i in self.target_object_ids:
            if self.object_names[i] in self.goal_description:
                self.goal_description[self.object_names[i]] += 1
            else:
                self.goal_description[self.object_names[i]] = 1

        room_type_path = f'./dataset/room_types.json'
        with open(room_type_path, 'r') as f: room_types = json.load(f)
        
        self.rooms_name = {}
        #now return <room_type> (id) for each room.        
        if type(layout) == str: now_layout = int(layout[0])
        else: now_layout = int(layout)
        for i, rooms_name in enumerate(room_types[scene[0]][now_layout]):
            if rooms_name not in ['Kitchen', 'Livingroom', 'Bedroom', 'Office']:
                the_name = None
            else:
                the_name = f'<{rooms_name}> ({1000 * (i + 1)})'
            self.rooms_name[i] = the_name

        self.done = False
        self.action_buffer = [[] for _ in range(self.number_of_agents)]

        resp = self.controller.communicate([{"$type": "send_scene_regions"}])
        self.scene_bounds = SceneBounds(resp=resp)
        self.all_rooms = [self.rooms_name[i] for i in range(len(self.rooms_name)) if self.rooms_name[i] is not None]
        info = {
            'goal_description': self.goal_description,
            'rooms_name': self.all_rooms,
            'agent_colors': self.replicant_colors,
        }
        env_api = [{
            'belongs_to_which_room': self.belongs_to_which_room,
            'center_of_room': self.center_of_room,
            'check_pos_in_room': self.check_pos_in_room,
            'get_room_distance': self.get_room_distance,
            'get_id_from_mask': partial(self.get_id_from_mask, agent_id=i),
            'get_with_character_mask': partial(self.get_with_character_mask, agent_id=i),
        } for i in range(self.number_of_agents)]
        self.obs = self.get_obs()
        return self.obs_filter(self.obs), info, env_api

    def pos_to_2d_box_distance(self, px, py, rx1, ry1, rx2, ry2):
        if px < rx1:
            if py < ry1:
                return ((px - rx1) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx1) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return rx1 - px
        elif px > rx2:
            if py < ry1:
                return ((px - rx2) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx2) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return px - rx2
        else:
            if py < ry1:
                return ry1 - py
            elif py > ry2:
                return py - ry2
            else:
                return 0
    
    def belongs_to_which_room(self, pos):
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis and self.rooms_name[i] is not None:
                min_dis = distance
                room = self.rooms_name[i]
        return room
    
    def get_room_distance(self, pos):
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis and self.rooms_name[i] is not None:
                min_dis = distance
                room = self.rooms_name[i]
        return min_dis
    
    def center_of_room(self, room):
        assert type(room) == str
        for index, name in self.rooms_name.items():
            if name == room:
                room = index
        return self.scene_bounds.regions[room].center
    
    def check_pos_in_room(self, pos):
        if len(pos) == 3:
            for region in self.scene_bounds.regions:
                if region.is_inside(pos[0], pos[2]):
                    return True
        elif len(pos) == 2:
            for region in self.scene_bounds.regions:
                if region.is_inside(pos[0], pos[1]):
                    return True
        return False

    def map_status(self, status, buffer_len = 0):
        if status == ActionStatus.ongoing or buffer_len > 0:
            return 0
        elif status == ActionStatus.success or status == ActionStatus.still_dropping:
            return 1
        else: return 2

    def get_2d_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[[0, 2]]) - np.array(pos2[[0, 2]]))

    def check_goal(self):
        r'''
        Check if the goal is achieved
        return: count, total, done
        '''
        place_pos = self.object_manager.transforms[self.goal_position_id].position
        count = 0
        for object_id in self.target_object_ids:
            pos = self.object_manager.transforms[object_id].position
            if (self.get_2d_distance(pos, place_pos) < 3 and self.belongs_to_which_room(pos) is not None and 'Bedroom' in self.belongs_to_which_room(pos)) or object_id in self.satisfied.keys():
                count += 1
                self.satisfied[object_id] = True
        return count, len(self.target_object_ids), count == len(self.target_object_ids)

    def get_id_from_mask(self, agent_id, mask, name = None):
        r'''
        Get the object id from the mask
        '''
        seg_with_mask = (self.obs[str(agent_id)]['seg_mask'] * np.expand_dims(mask, axis = -1)).reshape(-1, 3)
        seg_with_mask = [tuple(x) for x in seg_with_mask]
        seg_counter = Counter(seg_with_mask)
        
        for seg in seg_counter:
            if seg == (0, 0, 0): continue
            if seg_counter[seg] / np.sum(mask) > 0.5:
                for i in range(len(self.obs[str(agent_id)]['visible_objects'])):
                    if self.obs[str(agent_id)]['visible_objects'][i]['seg_color'] == seg:
                        return self.obs[str(agent_id)]['visible_objects'][i]
        return {
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                }

    def get_obs(self):
        for x in self.controller.state.containment.keys():
            if x not in self.containment_all.keys():
                self.containment_all[x] = []
            for y in self.controller.state.containment[x]:
                if y not in self.containment_all[x]:
                    self.containment_all[x].append(y)
        obs = {str(i): {} for i in range(self.number_of_agents)}
        containment_info_get = {str(i): [str(i)] for i in range(self.number_of_agents)}
        for replicant_id in self.controller.replicants:
            id = str(replicant_id)
            obs[id]['visible_objects'] = []
            if 'img' in self.controller.replicants[replicant_id].dynamic.images.keys():
                obs[id]['rgb'] = np.array(self.controller.replicants[replicant_id].dynamic.get_pil_image('img')).transpose(2, 0, 1)
                obs[id]['seg_mask'] = np.array(self.controller.replicants[replicant_id].dynamic.get_pil_image('id'))
                colors = Counter(self.controller.replicants[replicant_id].dynamic.get_pil_image('id').getdata())
                for object_id in self.segmentation_colors:
                    segmentation_color = tuple(self.segmentation_colors[object_id])
                    object_name = self.object_names[object_id]
                    if segmentation_color in colors:
                        obs[id]['visible_objects'].append({
                            'id': object_id,
                            'type': self.get_object_type(object_id),
                            'seg_color': segmentation_color,
                            'name': object_name,
                        })
                for agent_id in self.replicant_colors:
                    segmentation_color = tuple(self.replicant_colors[agent_id])
                    if segmentation_color in colors:
                        obs[id]['visible_objects'].append({
                            'id': agent_id,
                            'type': 3,
                            'seg_color': segmentation_color,
                            'name': 'agent',
                        })
                        if str(agent_id) not in containment_info_get[id]: containment_info_get[id].append(str(agent_id))
                        
                obs[id]['depth'] = np.flip(np.array(TDWUtils.get_depth_values(self.controller.replicants[replicant_id].dynamic.get_pil_image('depth'),
                        width = self.screen_size,
                        height = self.screen_size)), 0)
                obs[id]['camera_matrix'] = np.array(self.controller.replicants[replicant_id].dynamic.camera_matrix).reshape((4, 4))
            else:
                assert -1, "No image received"
            while len(obs[id]['visible_objects']) < 50:
                obs[id]['visible_objects'].append({
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                })
            x, y, z = self.controller.replicants[replicant_id].dynamic.transform.position
            fx, fy, fz = self.controller.replicants[replicant_id].dynamic.transform.forward
            obs[id]['agent'] = [x, y, z, fx, fy, fz]
            held_objects = list(self.controller.state.replicants[replicant_id].values())
            obs[id]['held_objects'] = []
            for hand in range(2):
                if held_objects[hand] is None:
                    obs[id]['held_objects'].append({
                        'id': None,
                        'type': None,
                        'name': None,
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                elif self.get_object_type(held_objects[hand]) == 0:
                    obs[id]['held_objects'].append({
                        'id': held_objects[hand],
                        'type': 0,
                        'name': self.object_names[held_objects[hand]],
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                else:
                    if held_objects[hand] in self.containment_all.keys():
                        contained_obj = [x for x in self.containment_all[held_objects[hand]] if x not in held_objects and x in self.target_object_ids]
                        obs[id]['held_objects'].append({
                            'id': held_objects[hand],
                            'type': 1,
                            'name': self.object_names[held_objects[hand]],
                            'contained': contained_obj + [None] * (3 - len(contained_obj)),
                            'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                        })
                    else:
                        obs[id]['held_objects'].append({
                            'id': held_objects[hand],
                            'type': 1,
                            'name': self.object_names[held_objects[hand]],
                            'contained': [None] * 3,
                            'contained_name': [None] * 3,
                        })
                        
            if len(containment_info_get[id]) == 2:
                oppo_held_objects = list(self.controller.state.replicants[1 - replicant_id].values())
            else:
                oppo_held_objects = [None, None]
            obs[id]['oppo_held_objects'] = []
            for hand in range(2):
                if oppo_held_objects[hand] is None:
                    obs[id]['oppo_held_objects'].append({
                        'id': None,
                        'type': None,
                        'name': None,
                        'contained': [None, None, None],
                        'contained_name': [None, None, None]
                    })
                elif self.get_object_type(oppo_held_objects[hand]) == 0:
                    obs[id]['oppo_held_objects'].append({
                        'id': oppo_held_objects[hand],
                        'type': 0,
                        'name': self.object_names[oppo_held_objects[hand]],
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                else:
                    if oppo_held_objects[hand] in self.containment_all.keys(): 
                        contained_obj = [x for x in self.containment_all[oppo_held_objects[hand]] if x not in oppo_held_objects and x in self.target_object_ids]
                        obs[id]['oppo_held_objects'].append({
                            'id': oppo_held_objects[hand],
                            'type': 1,
                            'name': self.object_names[oppo_held_objects[hand]],
                            'contained': contained_obj + [None] * (3 - len(contained_obj)),
                            'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                        })
                    else:
                       obs[id]['oppo_held_objects'].append({
                            'id': oppo_held_objects[hand],
                            'type': 1,
                            'name': self.object_names[oppo_held_objects[hand]],
                            'contained': [None] * 3,
                            'contained_name': [None] * 3,
                        })
            obs[id]['FOV'] = self.fov
            obs[id]['status'] = self.map_status(self.controller.replicants[replicant_id].action.status, len(self.action_buffer[replicant_id]))
            obs[id]['messages'] = [None, None]
            obs[id]['valid'] = True
            obs[id]['current_frames'] = self.num_frames
        return obs

    def get_info(self):
        #todo: add info needed
        return {}

    def add_name(self, inst):
        if type(inst) == int and inst in self.object_names:
            return f'{inst}_{self.object_names[inst]}'
        else:
            if type(inst) == dict:
                return {self.add_name(key): self.add_name(value) for key, value in inst.items()}
            elif type(inst) == list:
                return [self.add_name(item) for item in inst]
            else: raise NotImplementedError
    
    def add_name_and_empty(self, inst):
        for x in self.container_ids:
            if x not in inst:
                inst[x] = []
        return self.add_name(inst)

    def step(self, actions):
        '''
        Run one timestep of the environment's dynamics
        '''
        start = time.time()
        # Receive actions
        for replicant_id in self.controller.replicants:
            action = actions[str(replicant_id)]
            if action['type'] == 'ongoing': continue
            # otherwise we start an action directly
            self.action_buffer[replicant_id] = []
            if "arm" in action:
                if action['arm'] == 'left':
                    action['arm'] = Arm.left
                elif action['arm'] == 'right':
                    action['arm'] = Arm.right
            if action["type"] == 0:       # move forward 0.5m
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'move_forward'})
            elif action["type"] == 1:     # turn left by 15 degree
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'turn_left'})
            elif action["type"] == 2:     # turn right by 15 degree
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'turn_right'})
            elif action["type"] == 3:     # go to and grasp object with arm
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'reach_for'})
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'grasp'})
            elif action["type"] == 4:      # put in container
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'put_in'})
            elif action["type"] == 5:      # drop held object in arm
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'drop'})
            elif action["type"] == 6:      # send message
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'send_message'})
            else:
                assert False, "Invalid action type"

        # Do action here
        valid = [True for _ in range(self.number_of_agents)]
        delay_frame_count = [0 for _ in range(self.number_of_agents)]
        finish = False
        num_frames = 0
        while not finish: # continue until any agent's action finishes
            for replicant_id in self.controller.replicants:
                if delay_frame_count[replicant_id] > 0:
                    delay_frame_count[replicant_id] -= 1
                    continue
                if self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing and len(self.action_buffer[replicant_id]) == 0:
                    finish = True
                elif self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing:
                    curr_action = self.action_buffer[replicant_id].pop(0)
                    if curr_action['type'] == 'move_forward':       # move forward 0.5m
                        self.controller.replicants[replicant_id].move_forward()
                    elif curr_action['type'] == 'turn_left':     # turn left by 15 degree
                        self.controller.replicants[replicant_id].turn_by(angle = -15)
                    elif curr_action['type'] == 'turn_right':     # turn right by 15 degree
                        self.controller.replicants[replicant_id].turn_by(angle = 15)
                    elif curr_action['type'] == 'reach_for':     # go to and grasp object with arm
                        distance = self.get_2d_distance(self.controller.replicants[replicant_id].dynamic.transform.position, self.object_manager.transforms[int(curr_action["object"])].position)
                        if distance > self.reach_threshold:
                            valid[replicant_id] = False
                            self.action_buffer[replicant_id] = [] # the action is invaild
                        else: self.controller.replicants[replicant_id].move_to_position(self.object_manager.transforms[int(curr_action["object"])].position)
                    elif curr_action['type'] == 'grasp':
                        self.controller.replicants[replicant_id].grasp(int(curr_action["object"]), curr_action["arm"], relative_to_hand = False, axis = "yaw")
                    elif curr_action["type"] == 'put_in':      # put in container
                        self.controller.replicants[replicant_id].put_in()
                        held_objects = list(self.controller.state.replicants[replicant_id].values())
                        if held_objects[0] is not None and held_objects[1] is not None:
                            container, target = None, None
                            if self.get_object_type(held_objects[0]) == 1:
                                container = held_objects[0]
                            else:
                                target = held_objects[0]
                            if self.get_object_type(held_objects[1]) == 1:
                                container = held_objects[1]
                            else:
                                target = held_objects[1]
                            if container is not None and target is not None:
                                if container in self.containment_all:
                                    if target not in self.containment_all[container]:
                                        self.containment_all[container].append(target)
                                else:
                                    self.containment_all[container] = [target]
                    elif curr_action["type"] == 'drop':      # drop held object in arm
                        self.controller.replicants[replicant_id].drop(curr_action['arm'], max_num_frames = 30)
                    elif curr_action["type"] == 'send_message':      # send message
                        self.messages[replicant_id] = copy.deepcopy(curr_action['message'])
                        delay_frame_count[replicant_id] = max((len(self.messages[replicant_id]) - 1) // self.message_per_frame, 0)
            if finish: break
            data = self.controller.communicate([])
            for i in range(len(data) - 1):
                r_id = OutputData.get_data_type_id(data[i])
                if r_id == 'imag':
                    images = Images(data[i])
                    if images.get_avatar_id() == "a" and (self.num_frames + num_frames) % 1 == 0:
                        TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
            num_frames += 1

        self.num_frames += num_frames
        self.action_list.append(actions)
        goal_put, goal_total, self.success = self.check_goal()
        reward = 0
        for replicant_id in self.controller.replicants:
            action = actions[str(replicant_id)]
            task_status = self.controller.replicants[replicant_id].action.status
            self.f.write('step: {}, action: {}, time: {}, status: {}\n'
                    .format(self.num_step, action["type"],
                    time.time() - start,
                    task_status))
            container_info = self.add_name_and_empty(copy.deepcopy(self.controller.state.containment))
            self.f.write('position: {}, forward: {}, containment: {}, goal: {}, container: {}\n'.format(
                    self.controller.replicants[replicant_id].dynamic.transform.position,
                    self.controller.replicants[replicant_id].dynamic.transform.forward,
                    container_info, self.add_name(self.target_object_ids), self.add_name(self.container_ids)))
            self.f.flush()
            if task_status != ActionStatus.success and task_status != ActionStatus.ongoing:
                reward -= 0.1
        
        self.num_step += 1        
        self.reward += reward
        done = False
        if self.num_frames >= self.max_frame or self.success:
            done = True
            self.done = True
        
        obs = self.get_obs()
        # add messages to obs
        if self.number_of_agents == 2:
            for replicant_id in self.controller.replicants:
                obs[str(replicant_id)]['messages'] = copy.deepcopy(self.messages)
            self.messages = [None for _ in range(self.number_of_agents)]

        for replicant_id in self.controller.replicants:
            obs[str(replicant_id)]['valid'] = valid[replicant_id]
            obs[str(replicant_id)]['current_frames'] = self.num_frames

        info = self.get_info()
        info['done'] = done
        info['num_frames_for_step'] = num_frames
        info['num_step'] = self.num_step
        if done:
            info['reward'] = self.reward

        self.obs = obs
        return self.obs_filter(self.obs), reward, done, info
     
    def render(self):
        return None
        
    def save_images(self, save_dir='./Images'):
        '''
        save images of current step, including rgb, depth and segmentation image
        '''
        os.makedirs(save_dir, exist_ok=True)
        for replicant_id in self.controller.replicants:
            save_path = os.path.join(save_dir, str(replicant_id))
            os.makedirs(save_path, exist_ok=True)
            img = self.controller.replicants[replicant_id].dynamic.get_pil_image('img')
            depth = np.flip(np.array(TDWUtils.get_depth_values(self.controller.replicants[replicant_id].dynamic.get_pil_image('depth'), width = self.screen_size, height = self.screen_size), dtype = np.float32), 0)
            depth_img = Image.fromarray(100 / depth).convert('RGB')
            seg = self.controller.replicants[replicant_id].dynamic.get_pil_image('id')
            img.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}.png'))
            seg.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}_seg.png'))
            depth_img.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}_depth.png'))

    def close(self):
        print('close environment ...')
    #    with open(os.path.join(self.save_dir, 'action.pkl'), 'wb') as f:
    #        d = {'scene_info': self.scene_info, 'actions': self.action_list}
    #        pickle.dump(d, f)
