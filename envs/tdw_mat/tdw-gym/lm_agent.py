import json
import os
import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
import copy
from PIL import Image

from LLM.LLM import LLM

CELL_SIZE = 0.125
ANGLE = 15

def pos2map(x, z, _scene_bounds):
    i = int(round((x - _scene_bounds["x_min"]) / CELL_SIZE))
    j = int(round((z - _scene_bounds["z_min"]) / CELL_SIZE))
    return i, j



class lm_agent:
    def  __init__(self, agent_id, logger, max_frames, args, output_dir = 'results'):
        self.with_oppo = None
        self.oppo_pos = None
        self.with_character = None
        self.color2id = None
        self.satisfied = None
        self.object_list = None
        self.container_held = None
        self.gt_mask = None
        self.object_info = {} # {id: {id: xx, type: 0/1/2, name: sss, position: x,y,z}}
        self.object_per_room = {} # {room_name: {0/1/2: [{id: xx, type: 0/1/2, name: sss, position: x,y,z}]}}
        self.wall_map = None
        self.id_map = None
        self.object_map = None
        self.known_map = None
        self.occupancy_map = None
        self.agent_id = agent_id
        self.agent_type = 'lm_agent'
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 120)
        self.save_img = True
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.space_upd_freq = 30 # update spare space into the map
        self.logger = logger
        random.seed(1024)
        self.debug = True

        self.local_occupancy_map = None
        self.new_object_list = None
        self.visible_objects = None
        self.num_frames = None
        self.steps = None
        self.obs = None
        self.local_step = 0

        self.last_action = None
        self.pre_action = None

        self.goal_objects = None
        self.dropping_object = None

        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.plan = None

        self.rooms_name = None
        self.rooms_explored = {}
        self.position = None
        self.forward = None
        self.current_room = None
        self.holding_objects_id = None
        self.oppo_holding_objects_id = None
        self.oppo_last_room = None
        self.rotated = None


    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        depth = self.obs['depth'].copy()
        for i in range(len(self.obs['seg_mask'])):
            for j in range(len(self.obs['seg_mask'][0])):
                if (self.obs['seg_mask'][i][j] != color).any():
                    depth[i][j] = 1e9
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)

        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].copy().reshape(-1)
        yy = yy[index].copy().reshape(-1)
        depth = depth[index].copy().reshape(-1)

        pc = np.stack((xx, yy, depth, np.ones_like(xx)))

        pc = pc.reshape(4, -1)

        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)
        return rpc[:3]

    def cal_object_position(self, o_dict):
        pc = self.get_pc(o_dict['seg_color'])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]


    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def get_object_list(self):
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        for object_type in [0, 1, 2]:
            obj_map_indices = np.where(self.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.id_map[i, j]
                if id in self.satisfied or id in self.holding_objects_id or id in self.oppo_holding_objects_id or self.object_info[id] in object_list[object_type]:
                    continue
                object_list[object_type].append(self.object_info[id])
                room = self.env_api['belongs_to_which_room'](self.object_info[id]['position'])
                if room is None:
                    self.logger.warning(f"obj {self.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                self.object_per_room[room][object_type].append(self.object_info[id])
        self.object_list = object_list


    def get_new_object_list(self):
        self.visible_objects = self.obs['visible_objects']
        self.new_object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict['id'] is None: continue
            self.color2id[o_dict['seg_color']] = o_dict['id']
            if o_dict['id'] is None or o_dict['id'] in self.satisfied or o_dict['id'] in self.with_character or o_dict['type'] == 4:
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            object_id = o_dict['id']
            new_obj = False
            if object_id not in self.object_info:
                self.object_info[object_id] = {}
                new_obj = True
            self.object_info[object_id]['id'] = object_id
            self.object_info[object_id]['type'] = o_dict['type']
            self.object_info[object_id]['name'] = o_dict['name']
            if o_dict['type'] == 3: # the agent
                if o_dict['id'] == self.opponent_agent_id:
                    position = self.cal_object_position(o_dict)
                    self.oppo_pos = position
                    if position is not None:
                        oppo_last_room = self.env_api['belongs_to_which_room'](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                continue
            if object_id in self.satisfied or object_id in self.with_character:
                continue
            self.object_info[object_id]['position'] = position
            if o_dict['type'] == 0:
                x, y, z = self.object_info[object_id]['position']

                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[0].append(object_id)

            elif o_dict['type'] == 1:
                x, y, z = self.object_info[object_id]['position']
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[1].append(object_id)
            elif o_dict['type'] == 2:
                x, y, z = self.object_info[object_id]['position']
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[2].append(object_id)

    def color2id_fc(self, color):
        if color not in self.color2id:
            if color == (0, 0, 0):
                return -100 # wall
            else: return self.agent_id # agent
        else: 
            if self.color2id[color] in [0, 1]:
                return self.agent_id
            else: 
                return self.color2id[color]

    def dep2map(self):
        local_known_map = np.zeros_like(self.occupancy_map, np.int32)
        depth = self.obs['depth']

        filter_depth = depth.copy()
        for i in range(0, depth.shape[0]):
            for j in range(0, depth.shape[1]):
                if self.color2id_fc(tuple(self.obs['seg_mask'][i, j])) in self.with_character:
                    filter_depth[i, j] = 1e9
        depth = filter_depth
        #depth_img = Image.fromarray(100 / depth).convert('RGB')
        #depth_img.save(f'{self.output_dir}/Images/{self.agent_id}/{self.steps}_depth_filter.png')

        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)
        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        pc = np.stack((xx, yy, depth, np.ones((xx.shape[0], xx.shape[1]))))

        pc = pc.reshape(4, -1)

        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)

        rpc = np.dot(inv_E, pc).reshape(4, W, H)

        rpc = rpc.reshape(4, -1)
        depth = depth.reshape(-1)

        X = np.rint((rpc[0, :] - self._scene_bounds["x_min"]) / CELL_SIZE)
        X = np.maximum(X, 0)
        X = np.minimum(X, self.map_size[0] - 1)
        Z = np.rint((rpc[2, :] - self._scene_bounds["z_min"]) / CELL_SIZE)
        Z = np.maximum(Z, 0)
        Z = np.minimum(Z, self.map_size[1] - 1)

        index = np.where((depth > 0) & (depth < 5) & (rpc[1, :] < 1.5))
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        local_known_map[XX, ZZ] = 1

        # It may be necessary to remove the object from the occupancy map
        index = np.where((depth > 0) & (depth < 5) & (rpc[1, :] < 0.05)) # The object is moved, so the area remains empty, removing them from the occupancy map
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 0

        index = np.where((depth > 0) & (depth < 5) & (rpc[1, :] > 0.1) & (rpc[1, :] < 1.5)) # update the occupancy map
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 1
        self.local_occupancy_map[XX, ZZ] = 1

        index = np.where((depth > 0) & (depth < 5) & (rpc[1, :] > 2) & (rpc[1, :] < 3)) # it is a wall
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.wall_map[XX, ZZ] = 1
        return local_known_map

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def get_angle(self, forward, origin, position):
        p0 = np.array([origin[0], origin[2]])
        p1 = np.array([position[0], position[2]])
        d = p1 - p0
        d = d / np.linalg.norm(d)
        f = np.array([forward[0], forward[2]])

        dot = f[0] * d[0] + f[1] * d[1]
        det = f[0] * d[1] - f[1] * d[0]
        angle = np.arctan2(det, dot)
        angle = np.rad2deg(angle)
        return angle


    def reach_target_pos(self, target_pos, threshold = 1.0):
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        return d < threshold

    def conv2d(self, map, kernel=3):
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')

    def find_shortest_path(self, st, goal, map = None):
        st_x, _, st_z = st
        g_x, _, g_z = goal
        st_i, st_j = self.pos2map(st_x, st_z)
        g_i, g_j = self.pos2map(g_x, g_z)
        dist_map = np.ones_like(map, dtype=np.float32)
        super_map1 = self.conv2d(map, kernel=5)
        dist_map[super_map1 > 0] = 5
        super_map2 = self.conv2d(map)
        dist_map[super_map2 > 0] = 10
        dist_map[map > 0] = 50
        dist_map[self.known_map == 0] += 5
        dist_map[self.wall_map == 1] += 10000
        path = pyastar.astar_path(dist_map, (st_i, st_j),
                                  (g_i, g_j), allow_diagonal=False)
        return path

    def reset(self, obs, goal_objects = None, output_dir = None, env_api = None, rooms_name = None, gt_mask = True, save_img = True):
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self.oppo_pos = None
        goal_count = sum([v for k, v in goal_objects.items()])
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None

        #0: free, 1: occupied, 2: unknown
        self.occupancy_map = np.zeros(self.map_size, np.int32)
        #0: unknown, 1: known
        self.known_map = np.zeros(self.map_size, np.int32)
        #0: free, 1: target object, 2: container, 3: goal
        self.object_map = np.zeros(self.map_size, np.int32)
        #0: unknown; object_id(only target and container)
        self.id_map = np.zeros(self.map_size, np.int32)
        self.wall_map = np.zeros(self.map_size, np.int32)
        self.local_occupancy_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0
        print(self.obs.keys())
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)
        self.rotated = None
        self.rooms_explored = {}
        
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.gt_mask = gt_mask
        if self.gt_mask == True:
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection
            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
        self.navigation_threshold = 5
        print(self.rooms_name)
        self.LLM.reset(self.rooms_name, self.goal_objects)
        self.save_img = save_img

    def move(self, target_pos):
        self.local_step += 1
        local_known_map = self.dep2map()
        if self.local_step % self.space_upd_freq == 0:
            print("update local map")
            self.local_occupancy_map = copy.deepcopy(self.occupancy_map)
        self.known_map = np.maximum(self.known_map, local_known_map)
        path = self.find_shortest_path(self.position, target_pos, self.local_occupancy_map)
        i, j = path[min(5, len(path) - 1)]
        x, z = self.map2pos(i, j)
        angle = self.get_angle(forward=np.array(self.forward),
                               origin=np.array(self.position),
                               position=np.array([x, 0, z]))
        if np.abs(angle) < ANGLE:
            action = {"type": 0}
        elif angle > 0:
            action = {"type": 1}
        else:
            action = {"type": 2}
        return action


    def draw_map(self, previous_name):
        #DWH: draw the map
        draw_map = np.zeros((self.map_size[0], self.map_size[1], 3))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.occupancy_map[i, j] > 0:
                    draw_map[i, j] = 100
                if self.known_map[i, j] == 0:
                    assert self.occupancy_map[i, j] == 0
                    draw_map[i, j] = 50
                if self.wall_map[i, j] > 0:
                    draw_map[i, j] = 150
        draw_map[np.where(self.object_map == 1)] = [255, 0, 0]
        draw_map[np.where(self.object_map == 2)] = [0, 255, 0]
        draw_map[np.where(self.object_map == 3)] = [0, 0, 255]
        if self.oppo_pos is not None:
            draw_map[self.pos2map(self.oppo_pos[0], self.oppo_pos[2])] = [0, 255, 255]
        draw_map[self.pos2map(self.obs["agent"][0], self.obs["agent"][2])] = [255, 255, 0]
        #rotate the map 90 degrees anti-clockwise
        draw_map = np.rot90(draw_map, 1)
        cv2.imwrite(previous_name + '_map.png', draw_map)

    def gotoroom(self):
        target_room = ' '.join(self.plan.split(' ')[2: 4])
        if target_room[-1] == ',': target_room = target_room[:-1]
        print(target_room)
        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # add an interruption if anything new happens
        if len(self.new_object_list[0]) + len(self.new_object_list[1]) + len(self.new_object_list[2]) > 0:
            self.action_history[-1] = self.action_history[-1].replace(self.plan, f'go to {self.current_room}')
            self.new_object_list = {0: [], 1: [], 2: []}
            self.plan = None
            return None
        return self.move(target_pos)


    def goexplore(self):
        target_room = ' '.join(self.plan.split(' ')[-2:])
        # assert target_room == self.current_room, f"{target_room} != {self.current_room}"
        target_pos = self.env_api['center_of_room'](target_room)
        self.explore_count += 1
        dis_threshold = 1 + self.explore_count / 50
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)
        if self.rotated is None:
            self.rotated = 0
        if self.rotated == 16:
            self.roatated = 0
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            return None
        self.rotated += 1
        action = {"type": 1}
        return action

    def gograsp(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.object_map[np.where(self.id_map == target_object_id)] = 0
            self.id_map[np.where(self.id_map == target_object_id)] = 0
            self.plan = None
            return None
        
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_info[target_object_id]['position'])
        target_object_pos = self.target_pos

        if target_object_id not in self.object_info or target_object_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)
        action = {"type": 3, "object": target_object_id, "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'}
        return action
    
    def goput(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            self.with_character = [self.agent_id]
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_list[2][0]['position'])
        target_pos = self.target_pos

        if not self.reach_target_pos(target_pos, 1.5):
            return self.move(target_pos)
        if self.obs["held_objects"][0]['type'] is not None:
            self.dropping_object += [self.obs["held_objects"][0]['id']]
            if self.obs["held_objects"][0]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][0]['contained'] if x is not None]
            return {"type": 5, "arm": "left"}
        else:
            self.dropping_object += [self.obs["held_objects"][1]['id']]
            if self.obs["held_objects"][1]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][1]['contained'] if x is not None]
            return {"type": 5, "arm": "right"}

    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action
    
    def detect(self):
        detect_result = self.detection_model(self.obs['rgb'][..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((self.obs['rgb'].shape[0], self.obs['rgb'].shape[1], 3)).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3: continue
            mask = detect_result['masks'][:,:,i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](mask = mask, name = self.detection_model.cls_to_name_map(label)).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.with_character)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask

    def LLM_plan(self):
        return self.LLM.run(self.num_frames, self.current_room, self.rooms_explored, self.obs['held_objects'],[self.object_info[x] for x in self.satisfied if x in self.object_info], self.object_list, self.object_per_room, self.action_history, self.dialogue_history, self.obs['oppo_held_objects'], self.oppo_last_room)

    def act(self, obs):
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)
        self.num_frames = obs['current_frames']
        self.steps += 1

        if not self.gt_mask:
            self.obs['visible_objects'], self.obs['seg_mask'] = self.detect()

        if obs['valid'] == False:
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.satisfied.append(self.last_action['object'])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"
    
        if self.communication:
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}")
        if self.obs['status'] == 0: # ongoing
            return {'type': 'ongoing'}
    
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api['get_room_distance'](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'
        if self.agent_id not in self.with_character: self.with_character.append(self.agent_id) # DWH: buggy env, need to solve later.
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        for x in self.obs['held_objects']:
            if x['type'] == 0:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
            elif x['type'] == 1:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                #self.with_character.append(x['id'])
                for y in x['contained']:
                    if y is None:
                        break
                    if y not in self.with_character: self.with_character.append(y)
                    #self.with_character.append(y)
        oppo_name = {}
        oppo_type = {}
        for x in self.obs['oppo_held_objects']:
            if x['type'] == 0:
                self.oppo_holding_objects_id.append(x['id'])
                self.with_oppo.append(x['id'])
                oppo_name[x['id']] = x['name']
                oppo_type[x['id']] = x['type']
            elif x['type'] == 1:
                self.oppo_holding_objects_id.append(x['id'])
                self.with_oppo.append(x['id'])
                oppo_name[x['id']] = x['name']
                oppo_type[x['id']] = x['type']
                for i, y in enumerate(x['contained']):
                    if y is None:
                        break
                    self.with_oppo.append(y)
                    oppo_name[y] = x['contained_name'][i]
                    oppo_type[y] = 0
        for obj in self.with_oppo:
            if obj not in self.satisfied:
                self.satisfied.append(obj)
                self.object_info[obj] = {
                    "name": oppo_name[obj],
                    "id": obj,
                    "type": oppo_type[obj],
                }
                self.object_map[np.where(self.id_map == obj)] = 0
                self.id_map[np.where(self.id_map == obj)] = 0
        if not self.obs['valid']: # invalid, the object is not there
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
        if len(self.dropping_object) > 0 and self.obs['status'] == 1:
            self.logger.info("successful drop!")
            self.satisfied += self.dropping_object
            self.dropping_object = []
            self.plan = None

        self.get_new_object_list()
        print(self.new_object_list)
        self.get_object_list()

        info = {'satisfied': self.satisfied,
                'object_list': self.object_list,
                'new_object_list': self.new_object_list,
                'current_room': self.current_room,
                'visible_objects': self.filtered(self.obs['visible_objects']),
                'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
              }

        # save occupancy map:
        if self.save_img:
            self.draw_map(previous_name=f'{self.output_dir}/Images/{self.agent_id}/{self.steps:04}')

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                self.target_pos = None
                if lm_times > 0:
                    print(info)
                if lm_times > 3:
                    raise Exception(f"retrying LM_plan too many times")
                plan, a_info = self.LLM_plan()
                if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
                self.plan = plan
                self.action_history.append(f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}")
                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                lm_times += 1
            if self.plan.startswith('go to'):
                action = self.gotoroom()
            elif self.plan.startswith('explore'):
                self.explore_count = 0
                action = self.goexplore()
            elif self.plan.startswith('go grasp'):
                action = self.gograsp()
            elif self.plan.startswith('put'):
                action = self.putin()
            elif self.plan.startswith('transport'):
                action = self.goput()
            #    self.with_character = [self.agent_id]
            elif self.plan.startswith('send a message:'):
                action = {"type": 6,
                          "message": ' '.join(self.plan.split(' ')[3:])}
                self.plan = None
            elif self.plan.startswith('wait'):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        info.update({"action": action,
                     "plan": self.plan})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action
        return action


