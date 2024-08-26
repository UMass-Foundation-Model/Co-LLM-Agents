import os
import numpy as np
import cv2
import random
import time
import math
import copy
from agent_memory import AgentMemory

CELL_SIZE = 0.125
ANGLE = 15
MAX_GOAL_COUNT = 15

class H_agent:
    def __init__(self, agent_id, logger, max_frames, output_dir = 'results'):
        self.max_frames = max_frames
        self.env_api = None
        self.agent_id = agent_id
        self.agent_type = 'h_agent'
        self.local_step = 0
        self.is_reset = False
        self.logger = logger
        self.map_id = 0
        self.output_dir = output_dir
        self.last_action = None
        random.seed(1024)
        self.pre_action = None
        
        self.map_size = (240, 120)
        self.goal_objects = None
        self.goal_count = None
        self.drop_count = None
        self.gt_mask = None
        self.navigation_threshold = 5
        self.detection_threshold = 5
        self.save_img = True
        # detection_threshold < navigation_threshold when not gt_mask
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
    
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
        if pc.shape[1] < 5: return None
        position = pc.mean(1)
        return position[:3]

    def get_object_list(self):
        self.visible_objects = self.obs['visible_objects']
        self.object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict['id'] is None or o_dict['id'] in self.satisfied or o_dict['id'] in self.grasp:
                continue
            self.color2id[o_dict['seg_color']] = o_dict['id']
            object_id = o_dict['id']
            self.object_type[object_id] = o_dict['type']
            if o_dict['type'] > 3:
                continue
            if object_id in self.satisfied or object_id in self.grasp:
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            self.object_position[object_id] = position
            if o_dict['type'] == 0:
                x, y, z = self.get_object_position(object_id)                
                
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    self.object_list[0].append(object_id)
                    
            elif o_dict['type'] == 1:            
                x, y, z = self.get_object_position(object_id)
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    self.object_list[1].append(object_id)                    
            elif o_dict['type'] == 2:            
                x, y, z = self.get_object_position(object_id)
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    self.object_list[2].append(object_id)
            elif o_dict['type'] == 3:
                if object_id == self.agent_id: continue
                self.oppo_pos = self.get_object_position(object_id)

    def is_container(self, id):
        return id is not None and id in self.object_type and self.object_type[id] == 1
        
    def container_full(self, id):
        return self.content_container > 2
    
    def have_container(self):
        self.container_held = None
        for o in self.obs["held_objects"]:
            if o is not None and self.is_container(o):
                self.container_held = o
    
    def decide_sub(self):
        self.held_objects = self.obs["held_objects"]
        if self.num_frames > self.max_frames - 700 and (self.held_objects[0] is not None or self.held_objects[1] is not None):
            self.sub_goal = 2 # Need to go to the drop zone
        else:
            if self.goal_count == len(self.satisfied) + len(self.grasp) - self.drop_count - 1:
                self.sub_goal = 2
                return # the task is finished
            self.container_held = None
            for o in self.held_objects:
                if self.is_container(o):
                    self.container_held = o
                    if self.container_full(o):
                        self.sub_goal = 2
                    else:
                        self.sub_goal = 0
                    return
            if self.container_held is None:
                if self.held_objects[0] is not None and self.held_objects[1] is not None:
                    self.sub_goal = 2
                elif self.num_frames < self.max_frames - 1400:
                    self.sub_goal = 1 # can find a container first
                else:
                    self.sub_goal = 0
    
    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any(): 
                return -100 # wall
            else: return self.agent_id # agent
        else: return self.color2id[color]
        
    def get_object_position(self, object_id):
        return self.object_position[object_id]
    
    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5
    
    def check_goal(self, thresold = 1.0):  
        if self.goal is None:
            self.d = 0
            return False
        x, _, z = self.obs["agent"][:3]
        gx, gz = self.goal
        d = self.l2_distance((x, z), (gx, gz))
        self.d = d
        if 'belongs_to_which_room' in self.env_api and self.sub_goal == 2:
            flag = self.env_api['belongs_to_which_room'](np.array([x, 0, z])) == self.env_api['belongs_to_which_room'](np.array([gx, 0, gz]))
        else:
            flag = True
        return d < thresold and flag
    
    def Other_arm(self, arm):
        if arm == 'left':
            return 'right'
        else:
            return 'left'
    
    def ex_goal(self):
        try_step = 0
        while try_step < 100:
            try_step += 1
            goal = np.where(self.agent_memory.known_map == 0)
            idx = random.randint(0, goal[0].shape[0] - 1)
            i, j = goal[0][idx], goal[1][idx]
            if 'check_pos_in_room' in self.env_api:
                if self.env_api['check_pos_in_room'](self.map2pos(i, j)) == False:
                    continue
            if self.agent_memory.occupancy_map[i, j] == 0:
                self.goal = self.map2pos(i, j)
                return
        while True:
            try_step += 1
            goal = np.where(np.isin(self.agent_memory.known_map, [0, 1]))
            idx = random.randint(0, goal[0].shape[0] - 1)
            i, j = goal[0][idx], goal[1][idx]
            if 'check_pos_in_room' in self.env_api:
                if self.env_api['check_pos_in_room'](self.map2pos(i, j)) == False:
                    continue
            if self.agent_memory.occupancy_map[i, j] == 0:
                self.goal = self.map2pos(i, j)
                return
        
    def reset(self, goal_objects = None, max_frames = 3000, output_dir = None, env_api = {}, agent_color = [-1, -1, -1], agent_id = 0, gt_mask = True, save_img = True):
        self.is_reset = True
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.save_img = save_img
        self.force_ignore = []
        self.satisfied = []
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, gt_behavior=True, env_api=env_api, constraint_type = None, map_size = self.map_size, scene_bounds = self._scene_bounds)
        if goal_objects is not None:
            assert type(goal_objects) == dict
            self.goal_objects = goal_objects
            self.goal_count = sum([v for k, v in goal_objects.items()])
        else:
            self.goal_objects = None
            self.goal_count = MAX_GOAL_COUNT
        self.max_frames = max_frames
        if output_dir is not None:
            self.output_dir = output_dir
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
        print(self.goal_objects, self.goal_count)
    
    def _reset(self):
        self.W = self.map_size[0]
        self.H = self.map_size[1]
        self.last_action = None
        self.oppo_pos = None
        self.object_map = np.zeros(self.map_size, np.int32)
        #0: unknown; object_id(only target and container)
        self.id_map = np.zeros(self.map_size, np.int32)

        self.object_position = {}
        self.object_type = {}
        self.sub_goal = -1
        self.mode = None # nav, interact, move
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.space_upd_freq = 30 # update spare space into the map
        
        self.container_held = None
        self.grasp = []
        self.color2id = {}
        self.with_character = [self.agent_id]
        self.goal = None
        self.drop_count = 0
        
        self.num_step = 0
        self.num_frames = 0
        
        self.traj = []
        self.map_id = 0
        
        self.local_goal = None
        self.is_reset = False
        self.keep_drop = False
    
    def pre_interact(self, object_id):
        self.mode = 'move'
        self.interact_id = object_id
        x, y, z = self.get_object_position(object_id)
        self.goal = (x, z)
        if self.sub_goal == 2:
            self.move_d = 1.5
        else:
            self.move_d = 1
    
    def move(self):
        self.local_step += 1        
        action, path_len = self.agent_memory.move_to_pos(self.goal)
        return action
    
    def interact(self):
        if self.interact_mode == 'go_to':
            action = {"type": 3}
            action["object"] = self.interact_id
            return action
        if self.interact_mode == 'grasp':
            grasped_arm = []
            if self.obs["held_objects"][0] is None:
                grasped_arm.append('left')
            if self.obs["held_objects"][1] is None:
                grasped_arm.append('right')
            
            action = {"type": 3}
            action["object"] = self.interact_id
            action["arm"] = grasped_arm[0]
            self.grasp_arm = grasped_arm[0]
            return action
        if self.interact_mode == 'put_in':    
            action = {"type": 4,
                    "object": self.object_id,
                    "container": self.container_id}
            self.content_container += 1
            return action
        if self.interact_mode == 'drop':
            self.keep_drop = True
            if self.obs["held_objects"][0] is not None:
                return {"type": 5, "arm": "left"}
            else:
                return {"type": 5, "arm": "right"}

    
    def nav(self):
        self.local_step += 1
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        if len(self.object_list[1]) > 0:
            self.have_container()
            if self.sub_goal < 2 and self.container_held is None and self.num_frames < self.max_frames - 1400:
                self.sub_goal = 1
                goal = random.choice(self.object_list[1])                
                self.pre_interact(goal)                
                return
        if self.num_frames >= self.max_frames - 1400 and self.sub_goal < 2: self.sub_goal = 0
        if len(self.object_list[self.sub_goal]) > 0:                
            goal = random.choice(self.object_list[self.sub_goal])
            self.pre_interact(goal)
            return

        action, path_len = self.agent_memory.move_to_pos(self.goal)
        return action
    
    def update_grasp(self):
        self.object_map[np.where(self.id_map == self.interact_id)] = 0
        self.id_map[np.where(self.id_map == self.interact_id)] = 0        
        self.grasp.append(self.interact_id)
        self.with_character.append(self.interact_id)
        put_arm = self.Other_arm(self.grasp_arm)
        if self.is_container(self.interact_id):
            self.object_id = None
            self.content_container = 0
            if put_arm == 'left':
                self.object_id = self.obs["held_objects"][0]
            else:
                self.object_id = self.obs["held_objects"][1]
            if self.object_id is not None:
                self.interact_mode = "put_in"
                self.container_id = self.interact_id
                return
            else:
                self.sub_goal = -1
        else:
            self.have_container()
            if self.container_held is not None:
                self.interact_mode = "put_in"
                self.container_id = self.container_held
                self.object_id = self.interact_id
                return
            else:
                self.sub_goal = -1

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
        #self.detection_model(self.obs['rgb'][..., [2, 1, 0]], decode = False, no_save_pred = False, out_dir = 'outputs')
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.with_character)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask

    def act(self, obs):
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)
        if self.is_reset:
            self._reset()

        if not self.gt_mask:
            self.obs['visible_objects'], self.obs['seg_mask'] = self.detect()

        ignore_obstacles = []
        ignore_ids = []
        self.with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x is None or x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        for x in self.force_ignore:
            self.with_character.append(x)

        for x in self.obs["oppo_held_objects"]:
            if x is None or x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        self.agent_memory.update(
            obs, ignore_ids=ignore_ids, ignore_obstacles=ignore_obstacles, save_img = self.save_img
        )

        self.num_frames = obs['current_frames']
        self.num_step += 1
        self.obs['held_objects'] = [self.obs['held_objects'][0]['id'], self.obs['held_objects'][1]['id']]

        other_agents_objects = []
        for hand in range(2):
            if self.obs['oppo_held_objects'][hand]['id'] != None:
                other_agents_objects.append(self.obs['oppo_held_objects'][hand]['id'])
            for contained_id in self.obs['oppo_held_objects'][hand]['contained']:
                if contained_id is not None:
                    other_agents_objects.append(contained_id)
                    
        for obj in other_agents_objects:
            if obj not in self.satisfied:
                self.satisfied.append(obj)
                self.object_map[np.where(self.id_map == obj)] = 0
                self.id_map[np.where(self.id_map == obj)] = 0

        if self.obs['status'] == 0: # ongoing, only update the map, do not act.
            return {'type': 'ongoing'}
        if self.obs['valid'] == False: # invalid, the object is not there
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
                
        self.get_object_list()

        if self.keep_drop:
            if self.obs["held_objects"][0] is not None:
                self.last_action = {"type": 5, "arm": "left"}
                return {"type": 5, "arm": "left"}
            elif self.obs["held_objects"][1] is not None:
                self.last_action = {"type": 5, "arm": "right"}
                return {"type": 5, "arm": "right"}
            else:
                self.satisfied += self.with_character[1:]
                self.drop_count += 1
                self.grasp = []
                self.with_character = [self.agent_id] #the repicant
                self.keep_drop = False
            #DWH: modify the agent to fit new env.

        if self.sub_goal != -1:
            if self.mode == 'nav':
                if self.local_step >= self.max_nav_steps \
                        or self.check_goal(2.0):
                    self.sub_goal = -1
                    
            elif self.mode == 'move':
                if self.check_goal(self.move_d):                    
                    self.mode = 'interact'
                    if self.sub_goal < 2:
                        self.interact_mode = 'grasp' #'go_to'
                        self.try_step = 0
                    else:
                        self.interact_mode = 'drop'
                elif self.local_step >= self.max_move_steps:
                    self.sub_goal = -1
                
            elif self.mode == 'interact':
                if self.interact_mode == "grasp":
                    if self.obs['status'] == 1:
                        self.update_grasp()                        
                    elif self.try_step == 2:
                        self.sub_goal = -1
                    else:
                        self.try_step += 1
                else:
                    self.sub_goal = -1
        
        if self.sub_goal == -1:
            self.decide_sub()
            self.local_step = 0            
            goal = np.where(self.object_map == self.sub_goal + 1)
            if goal[0].shape[0] > 0:
                idx = 0
                idx = random.randint(0, len(goal[0]) - 1)
                i, j = goal[0][idx], goal[1][idx]                
                self.pre_interact(self.id_map[i, j])
            else:
                self.ex_goal()                
                self.mode = 'nav'
        if self.mode == 'nav':
            #self.logger.debug("Navigating ...")
            action = self.nav()
            #self.logger.debug("Got Navigating action")
            if self.mode == 'nav':
                self.last_action = action
                return action
            else:
                self.local_step = 0
        if self.mode == 'move':
            #self.logger.debug("Moving ...")
            action = self.move()
            #self.logger.debug("Got Moving ...")
            self.last_action = action
            return action

        if self.mode == 'interact':
            #self.logger.debug("Interacting ...")
            action = self.interact()
            self.logger.debug(action)
            #self.logger.debug("Got Interaction action")
            self.last_action = action
            return action
