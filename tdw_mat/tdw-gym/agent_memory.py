import numpy as np
import math
import random
import pyastar2d as pyastar
from PIL import Image
import copy
import cv2
import os
import time
from collections import defaultdict

CELL_SIZE = 0.125
ANGLE = 15
IGNORE_TIME = 500
WALL_HEIGHT = 2
TRUE_WALL_HEIGHT = 2.9
SURROUNDING_DISTANCE = 1.5
CIRCLE_RADIUS = 0.15
SPACE_UPDATE_FREQ = 10
WARMUP_EXPLORE_FRAMES = 200
EXPLORE_MAX_COST = 10000
MINIMAL_EXPLORE_STEP = 100
BEHAVIOR_DETECT_INTERVAL = 15
#Some wall may be detected as object, so we need to remove them from the wall map
class AgentMemory():
    def __init__(self, agent_id, agent_color, output_dir = None, gt_mask = False, gt_behavior = False, env_api = None, constraint_type = None, map_size = None, scene_bounds = None):
        self.map_size = map_size
        self.env_api = env_api
        self.turn_around_count = 0
        self.goal_objects = None
        self._scene_bounds = scene_bounds
        self.obs = None        
        #0: free, 1: occupied, 2: unknown
        #occupancy map is the map that the agent that has been explored, updated every frames
        #The difference of occupancy map and local occupancy map is that 
        #the occupancy map that local occupancy map does not update free space,
        #but copy the occupancy map every 30 frames
        self.occupancy_map = np.zeros(self.map_size, np.int32)
        self.local_occupancy_map = np.zeros(self.map_size, np.int32)
        #0: unknown, 1: known
        self.known_map = np.zeros(self.map_size, np.int32)
        #0: free, 1: target object, 2: container, 3: goal
        self.object_map = np.zeros(self.map_size, np.int32)
        #0: unknown; object_id(only target and container)
        self.current_obstacle_map = np.zeros_like(self.occupancy_map, np.float32)
        self.id_map = np.zeros(self.map_size, np.int32)
        self.wall_map = np.zeros(self.map_size, np.int32)
        self.true_wall_map = np.zeros(self.map_size, np.int32)
        self.local_step = 0
        self.agent_id = agent_id
        self.agent_color = agent_color
        self.ignore_ids = []
        self.ignore_obstacles = []
        #Update spare space into the local occupancy map every 30 frames
        self.object_info = {}
        self.color2id = {}
        self.output_dir = output_dir
        self.oppo_pos = None
        self.third_agent_pos = None
        self.ignore_id_times = {}
        self.ignore_obstacle_times = {}
        self.target = None
        self.gt_mask = gt_mask
        self.gt_behavior = gt_behavior
        self.detection_threshold = 5
        self.navigation_threshold = 5
        self.oppo_this_step = False
        self.third_agent_this_step = False
        self.constraint_type = constraint_type
        self.invalid_counter = dict()
        self.count = 0
        self.oppo_frame = 0
        self.third_agent_frame = 0
        if self.gt_mask == False:
            from detection import init_detection
            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
            self.detection_threshold = 3

        if self.gt_behavior == False:
            from behavior import init_behavior
            # only here we need to use the behavior model, other places we use the gt mask
            # so we put the import here
            self.behavior_model = init_behavior()
            self.last_behavior_detect = 0
            
        #! Keep track of action history and status history
        self.action_history_dict = {0: [None], 1: [None], 2: [None]}
        self.status_history_dict = {0: [None], 1: [None], 2: [None]}
        
        #! Keep track of oppo held objects history
        self.oppo_held_objects_history = [
            [{'id': None, 'type': None, 'name': None, 'contained': [None, None, None], 'contained_name': [None, None, None]}, 
             {'id': None, 'type': None, 'name': None, 'contained': [None, None, None], 'contained_name': [None, None, None]}]
        ]

    def detect(self, rgb):
        detect_result = self.detection_model(rgb[..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((rgb.shape[0], rgb.shape[1], 3)).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3: continue
            mask = detect_result['masks'][:,:,i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](mask = mask, name = self.detection_model.cls_to_name_map(label)).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.ignore_ids)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask

    def behavior(self, img_list_len = 50):
        assert self.agent_id == 1
        if not self.oppo_this_step:
            return None
        image_path = os.path.join(self.output_dir, 'Images', '1')
        img_list_all = [os.path.join(image_path, x) for x in os.listdir(image_path) if x[:-4].isdigit()]
        img_list_len = min(img_list_len, len(img_list_all))
        img_list = sorted(img_list_all)[-img_list_len:]
        result = self.behavior_model(img_list)
        with open("behavior.txt", "a") as f:
            f.write(f"{self.obs['current_frames']} {result}\n")

        if result[2] < 0.6 or result[0] > 5:
            return None
        success = result[0] % 2
        if result[0] in [0, 1]:
            action_type = 3
        if result[0] in [2, 3]:
            action_type = 4
        if result[0] in [4, 5]:
            action_type = 5
        action_name = self.merge_action_and_location(action_type)
        return action_name, success

    def ignore_logic(self, current_frames, ignore_ids = [], ignore_obstacles = []):
        # Ignore logic: if the object is not there, remove it from the memory and do not add it in the future 500 frames
        for ignore_id in ignore_ids:
            self.ignore_id_times[ignore_id] = current_frames
        for ignore_obstacle in ignore_obstacles:
            self.ignore_obstacle_times[ignore_obstacle] = current_frames
        self.ignore_ids += ignore_ids # remove them from memory
        self.ignore_obstacles += ignore_obstacles # remove them from occupancy map (only this step)
        # Unique:
        self.ignore_ids = list(set(self.ignore_ids))
        self.ignore_obstacles = list(set(self.ignore_obstacles))
        # Remove from memory:
        for ignore_id in self.ignore_ids: 
            # the object is not there or satisfied, or with the agent
            self.object_map[np.where(self.id_map == ignore_id)] = 0
            self.id_map[np.where(self.id_map == ignore_id)] = 0
        # Clear the ignore list:
        for ignore_id in self.ignore_ids:
            if current_frames - self.ignore_id_times[ignore_id] > IGNORE_TIME:
                self.ignore_ids.remove(ignore_id)
                if ignore_id in self.invalid_counter:
                    self.invalid_counter[ignore_id] = 0
                    
        for ignore_obstacle in self.ignore_obstacles:
            if current_frames - self.ignore_obstacle_times[ignore_obstacle] > IGNORE_TIME:
                self.ignore_obstacles.remove(ignore_obstacle)
    
    def update(self, obs, ignore_ids = [], ignore_obstacles = [], save_img = False):
        self.obs = obs
        self.ignore_logic(self.obs['current_frames'], ignore_ids, ignore_obstacles)
        self.local_step += 1
        self.get_object_list()
        local_known_map = self.dep2map()
        self.known_map = np.maximum(self.known_map, local_known_map)
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        if self.local_step % SPACE_UPDATE_FREQ == 0:
            self.local_occupancy_map = copy.deepcopy(self.occupancy_map)
        if save_img:
            os.makedirs(os.path.join(self.output_dir, 'Images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'Images', str(self.agent_id)), exist_ok=True)
            self.draw_map(previous_name = os.path.join(self.output_dir, 'Images', str(self.agent_id), f'{self.obs["current_frames"]:04}'))

        # Behavior Detection
        if not self.gt_behavior and self.obs['current_frames'] > self.last_behavior_detect + BEHAVIOR_DETECT_INTERVAL:
            behavior = self.behavior()
            self.last_behavior_detect = self.obs['current_frames']
            if behavior is not None:
                action_name, success = behavior
                self.obs['previous_action'][self.agent_id] = action_name
                self.obs['previous_status'][self.agent_id] = 'success' if success == 0 else 'fail'
            
        # Update action history and status history
        if int(self.agent_id) != 2 and 'previous_action' in self.obs:
            for agent_id in self.obs['previous_action'].keys():
                previous_action = self.obs['previous_action'][agent_id]
                previous_status = self.obs['previous_status'][agent_id]
                last_action = self.action_history_dict[agent_id][-1]
                last_status = self.status_history_dict[agent_id][-1]
                if last_action != previous_action and previous_action != None:
                    self.action_history_dict[agent_id].append(previous_action)
                    self.status_history_dict[agent_id].append(previous_status)
                elif last_action == previous_action and previous_status != last_status and previous_status != None:
                    self.status_history_dict[agent_id][-1] = previous_status
        
        # Update oppo held objects history
        if int(self.agent_id) != 2:
            if any(obj['type'] is not None for obj in self.obs['oppo_held_objects']):
                self.oppo_held_objects_history.append(self.obs['oppo_held_objects'])

    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j
        
    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z
    
    def get_pc(self, color) -> np.ndarray:
        depth = self.obs['depth'].copy()
        mask = np.any(self.obs['seg_mask'] != color, axis=-1)
        depth[mask] = 1e9
        # Camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))
        
        # Ego
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
            return None, None
        
        position = pc.mean(1)
        return position[:3], pc
    
    def get_object_list(self):
        self.oppo_this_step = False
        self.third_agent_this_step = False
        self.visible_objects = self.obs['visible_objects']
        for o_dict in self.visible_objects:
            if o_dict['id'] is None:
                continue
            self.color2id[o_dict['seg_color']] = o_dict['id']
            if o_dict['id'] in self.ignore_ids:
                continue
            object_id = o_dict['id']
            position, pc = self.cal_object_position(o_dict)

            if position is None:
                # Too far from the agent
                continue
            self.object_info[o_dict['id']] = o_dict
            self.object_info[o_dict['id']]['position'] = position
            x, y, z = self.get_object_position(object_id)
            i, j = self.pos2map(x, z)
            if i < 0 or i >= self.object_map.shape[0] or j < 0 or j >= self.object_map.shape[1]:
                continue
            # WARNING: Be careful to change to code, the object_map is 1-based
            # WARNING: type 3 is agent, exclude it
            if self.object_map[i, j] == 0 and o_dict['type'] != 3 and not (o_dict['type'] == 1 and (self.constraint_type is not None and "wheelchair" in self.constraint_type)):
                self.object_map[i, j] = o_dict['type'] + 1
                self.id_map[i, j] = object_id
            if o_dict['type'] == 3:
                if object_id == self.agent_id: continue
                if object_id == 2:
                    self.third_agent_pos = self.get_object_position(object_id)
                    self.third_agent_this_step = True
                    self.third_agent_frame = self.obs['current_frames']
                else:
                    self.oppo_pos = self.get_object_position(object_id)
                    self.oppo_this_step = True
                    self.oppo_frame = self.obs['current_frames']
        
        if self.agent_id == 0:
            # the first agent cannot go through the obstacle, so add them to the obstacle map
            self.current_obstacle_map = np.zeros_like(self.occupancy_map, np.float32)
            self.current_obstacle_map[np.where(self.object_map == 5)] = self.id_map[np.where(self.object_map == 5)]
            self.current_obstacle_map = self.conv2d(self.current_obstacle_map, kernel=12)
            # with open("file.txt", "a") as f:
            #     idxs = np.where(self.object_map == 5)
            #     for i in range(len(idxs[0])): 
            #         x, y = idxs[0][i], idxs[1][i]
            #         current_frame = self.obs["current_frames"]
            #         f.write(f"{current_frame} {x} {y} {self.id_map[x, y]} {self.current_obstacle_map[x, y]}\n")

        # the second agent can go through the obstacle, so remove them from the obstacle map
        obstacle_map = np.zeros_like(self.occupancy_map, np.float32)
        obstacle_map[np.where(self.object_map == 4)] = 1
        obstacle_map = self.conv2d(obstacle_map, kernel=5)
        self.occupancy_map[np.where(obstacle_map > 0)] = 0
        self.local_occupancy_map[np.where(obstacle_map > 0)] = 0

    
    def color2id_fc_vectorized(self, seg_mask):
        # Flatten the seg_mask for vectorized operation
        flat_seg_mask = seg_mask.reshape(-1, seg_mask.shape[-1])
        # Convert colors to a tuple for dictionary key lookups
        colors = list(map(tuple, flat_seg_mask))

        # Prepare an array to store the results
        color_ids = np.full(flat_seg_mask.shape[0], -100)  # default to wall or opposite agent

        # Check for agent color and assign the agent ID
        is_agent = np.all(flat_seg_mask == self.agent_color, axis=1)
        color_ids[is_agent] = self.agent_id

        # Process other colors
        for idx, color in enumerate(colors):
            if color in self.color2id:
                color_ids[idx] = self.color2id[color]

        return color_ids.reshape(seg_mask.shape[:2])

    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any():
                # Wall or opposite agent
                return -100
            else:
                # agent
                return self.agent_id
        else:
            return self.color2id[color]
    def dep2map(self):
        local_known_map = np.zeros_like(self.occupancy_map, np.int32)
        depth = self.obs['depth']

        color_ids = self.color2id_fc_vectorized(self.obs['seg_mask'])
        # Create a mask for ignored obstacles
        # with open("file.txt", "a") as f:
        #     f.write(f"current dep2map ignore: {self.obs['current_frames']} {self.ignore_obstacles}\n\n")
        ignore_mask = np.isin(color_ids, self.ignore_obstacles)
            
        filter_depth = np.where(ignore_mask, 1e9, np.zeros_like(depth))
        # Since the depth maybe inaccurate, move the depth near mask boundary to the mask boundary
        mask_bound = np.where(self.conv2d(color_ids, kernel=3) - color_ids * 9 != 0)
        filter_depth[mask_bound] = 1e9

        filter_depth = self.conv2d(filter_depth, kernel=5)
        depth = np.maximum(depth, filter_depth)

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
        
        index = np.where((depth > 0) & (depth < self.detection_threshold) & (rpc[1, :] < 2 * WALL_HEIGHT))
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        local_known_map[XX, ZZ] = 1

        # It may be necessary to remove the object from the occupancy map
        index = np.where((depth > 0) & (depth < self.navigation_threshold) & (rpc[1, :] < 0.05)) 
        # The object is moved, so the area remains empty, removing them from the occupancy map
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 0
        
        index = np.where((depth > 0) & (depth < self.navigation_threshold) & (rpc[1, :] > 0.1) & (rpc[1, :] <= WALL_HEIGHT)) 
        # update the occupancy map
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 1
        self.local_occupancy_map[XX, ZZ] = 1
        
        index = np.where((depth > 0) & (depth < self.navigation_threshold) & (rpc[1, :] < 0.05)) 
        # Some object may be recognized as wall, so we need to remove them from the wall map
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        removed_index = np.where(self.local_occupancy_map[XX, ZZ] == 0)
        XX = XX[removed_index]
        ZZ = ZZ[removed_index]
        self.wall_map[XX, ZZ] = 0

        index = np.where((depth > 0) & (depth < 5) & (rpc[1, :] > WALL_HEIGHT) & (rpc[1, :] < 2 * WALL_HEIGHT)) # it is a wall
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.wall_map[XX, ZZ] = 1

        index = np.where((depth > 0) & (depth < 5) & (rpc[1, :] > TRUE_WALL_HEIGHT) & (rpc[1, :] < 2 * WALL_HEIGHT)) # it is a wall
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.true_wall_map[XX, ZZ] = 1

        if np.all(depth < 0.375):
            # The agent is too close to the wall, so we need to add the pos in front of the agent to the wall map
            x, z = self.position[0], self.position[2]
            direction = np.array(self.forward)
            x += direction[0] * 0.25
            z += direction[2] * 0.25
            i, j = self.pos2map(x, z)
            self.wall_map[i, j] = 1
        return local_known_map
        
    def get_object_position(self, object_id):
        return self.object_info[object_id]['position']
    
    def l2_distance(self, st, g):
        if len(st) == 2:
            st_x, st_z = st
        elif len(st) == 3:
            st_x, _, st_z = st
        else:
            raise ValueError("Start format not supported!")
        if len(g) == 2:
            g_x, g_z = g
        elif len(g) == 3:
            g_x, _, g_z = g
        else:
            raise ValueError("Goal format not supported!")
        return np.sqrt((st_x - g_x) ** 2 + (st_z - g_z) ** 2)
    
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
        #with open("path.txt", "a") as file:
        #    file.write(f"{origin[0]} {origin[2]} {position[0]} {position[2]} {f[0]} {f[1]} {d[0]} {d[1]} {dot} {det} {angle}\n")

        return angle
    
    def conv2d(self, map, kernel=3):
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')
    
    def find_shortest_path(self, st, goal):
        if len(st) == 2:
            st_x, st_z = st
        elif len(st) == 3:
            st_x, _, st_z = st
        else:
            raise ValueError("Start format not supported!")
        if len(goal) == 2:
            g_x, g_z = goal
        elif len(goal) == 3:
            g_x, _, g_z = goal
        else:
            raise ValueError("Goal format not supported!")
        st_i, st_j = self.pos2map(st_x, st_z)
        g_i, g_j = self.pos2map(g_x, g_z)
        dist_map = np.ones_like(self.local_occupancy_map, dtype=np.float32)
        super_map1 = self.conv2d(self.local_occupancy_map, kernel=5)
        dist_map[super_map1 > 0] = 5
        super_map2 = self.conv2d(self.local_occupancy_map, kernel=3)
        dist_map[super_map2 > 0] = 10
        super_map3 = self.conv2d(self.wall_map, kernel=1)
        dist_map[super_map3 > 0] = 10000
        if self.obs['current_frames'] < WARMUP_EXPLORE_FRAMES:
            dist_map[self.local_occupancy_map > 0] = 500000
        else:
            dist_map[self.local_occupancy_map > 0] = 1000
        dist_map[self.known_map == 0] += 2.5
        dist_map[self.wall_map > 0] += 500000
        dist_map[self.true_wall_map > 0] += 5000000
        dist_map[self.current_obstacle_map > 0] += 200000
        path = pyastar.astar_path(dist_map, (st_i, st_j), (g_i, g_j), allow_diagonal=False)
        # Calculate the distance
        distance = 0
        for i in range(1, len(path)):
            # Ignore the first grid
            distance += dist_map[path[i][0], path[i][1]]
        return path, distance
    
    def have_wall(self, st_pos, de_pos):
        for i in range(min(st_pos[0], de_pos[0]), max(st_pos[0], de_pos[0])):
            for j in range(min(st_pos[1], de_pos[1]), max(st_pos[1], de_pos[1])):
                if self.wall_map[i, j] > 0:
                    return True
        return False

    def move_to_pos(self, pos, explore = False, follow = False, follow_main_agent = False, follow_third_agent = False, nav_step = 0):
        local_known_map = self.dep2map()
        self.known_map = np.maximum(self.known_map, local_known_map)
        path, cost = self.find_shortest_path(self.position, pos)
        i, j = path[min(5, len(path) - 1)]
        if self.have_wall(path[0], path[min(5, len(path) - 1)]):
            i, j = path[min(1, len(path) - 1)]

        self.target = path[-1]
        x, z = self.map2pos(i, j)

        angle = self.get_angle(forward=np.array(self.forward),
                            origin=np.array(self.position),
                            position=np.array([x, 0, z]))
        action = None
        if follow == True:
            if self.oppo_this_step == True:
                self.turn_around_count = 0
            if self.l2_distance(self.position, pos) < 3 and self.oppo_this_step == True:
                angle = self.get_angle(forward=np.array(self.forward),
                        origin=np.array(self.position),
                        position=np.array(pos))
                if angle > ANGLE * 0.75:
                    action = {"type": 1}
                elif angle < - ANGLE * 0.75:
                    action = {"type": 2}
                else:
                    action = {"type": 8, "delay": 2}
            if self.l2_distance(self.position, pos) < 0.5 and self.oppo_this_step == False:
                if self.turn_around_count < 270 / ANGLE:
                    action = {"type": 1}
                    self.turn_around_count += 1
                else:
                    self.turn_around_count = 0
                    self.oppo_pos = None
                self.oppo_pos = None
        if follow_main_agent == True:
            if self.l2_distance(self.position, pos) < 2:
                angle = self.get_angle(forward=np.array(self.forward),
                        origin=np.array(self.position),
                        position=np.array(pos))
                if angle > ANGLE * 0.75:
                    action = {"type": 1}
                elif angle < - ANGLE * 0.75:
                    action = {"type": 2}
                else:
                    action = {"type": 8, "delay": 1}

        if follow_third_agent == True:
            if self.l2_distance(self.position, pos) < 2 and self.third_agent_this_step == True:
                angle = self.get_angle(forward=np.array(self.forward),
                            origin=np.array(self.position),
                            position=np.array(pos))
                if angle > ANGLE * 0.75:
                    action = {"type": 1}
                elif angle < - ANGLE * 0.75:
                    action = {"type": 2}
                else:
                    action = {"type": 8, "delay": 1}
            if self.l2_distance(self.position, pos) < 0.5 and self.third_agent_this_step == False:
                # Not there any more
                self.third_agent_pos = None

        if action is None:
            if np.abs(angle) < ANGLE * 1.2:
                action = {"type": 0}
            elif angle > 0:
                action = {"type": 1}
            else:
                action = {"type": 2}

        if explore:
            if self.l2_distance(self.position, pos) < SURROUNDING_DISTANCE or (self.known_map[self.pos2map(pos[0], pos[-1])] != 0 and nav_step > MINIMAL_EXPLORE_STEP) or cost > EXPLORE_MAX_COST:
                action = None
        # Calcuate cost:
        if cost > 500000 and self.agent_id == 0 and self.constraint_type is not None and "wheelchair" in self.constraint_type and self.wall_map[self.pos2map(pos[0], pos[-1])] == 0:
            # Wait for the obstacle to move
            action = {"type": 1}

        # print("now save map: ", os.path.join(self.output_dir, 'Images', str(self.agent_id), f'{self.obs["current_frames"]:04}'))
        return action, len(path) 
    
    def draw_map(self, previous_name, save = True):
        # Draw the map
        #with open("map.txt", "a") as f:
        #    for i in range(self.map_size[0]):
        #        for j in range(self.map_size[1]):
        #            f.write(f"{self.occupancy_map[i, j]} ")
        #        f.write("\n")
        #    f.write("\n")

        draw_map = np.zeros((self.map_size[0], self.map_size[1], 3))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.occupancy_map[i, j] > 0:
                    draw_map[i, j] = 100
                if self.known_map[i, j] == 0:
                    draw_map[i, j] = 50
                if self.wall_map[i, j] > 0 or self.current_obstacle_map[i, j] > 0:
                    draw_map[i, j] = 150
                if self.true_wall_map[i, j] > 0:
                    draw_map[i, j] = 200

        draw_map[np.where(self.object_map == 1)] = [255, 0, 0]
        draw_map[np.where(self.object_map == 2)] = [0, 255, 0]
        draw_map[np.where(self.object_map == 3)] = [0, 0, 255]
        draw_map[np.where(self.object_map == 4)] = [127, 255, 255]
        if self.oppo_pos is not None:
            draw_map[self.pos2map(self.oppo_pos[0], self.oppo_pos[2])] = [0, 255, 255]
        if self.third_agent_pos is not None:
            draw_map[self.pos2map(self.third_agent_pos[0], self.third_agent_pos[2])] = [0, 127, 127]

        draw_map[self.pos2map(self.obs["agent"][0], self.obs["agent"][2])] = [255, 255, 0]
        if self.target is not None:
            draw_map[self.target[0], self.target[1]] = [255, 0, 255]
        #rotate the map 90 degrees anti-clockwise
        draw_map = np.rot90(draw_map, 1)
        if save == True:
            cv2.imwrite(previous_name + '_map.png', draw_map)
        return draw_map

    def visualize_depth_filter(self):
        pass
        #TODO: not implemented
        # depth_img = Image.fromarray(100 / depth).convert('RGB')
        # depth_img.save(f'{self.output_dir}/Images/{self.agent_id}/{self.num_step}_depth_filter.png')

    def sum_circle(self, map, i, j, radius):
        sum = 0
        for x in range(-int(radius / CELL_SIZE), int(radius / CELL_SIZE) + 1):
            for y in range(-int(radius / CELL_SIZE), int(radius / CELL_SIZE) + 1):
                if i + x < 0 or i + x >= map.shape[0] or j + y < 0 or j + y >= map.shape[1]:
                    continue
                if (x * CELL_SIZE) ** 2 + (y * CELL_SIZE) ** 2 > radius ** 2:
                    continue
                sum += map[i + x, j + y]
        return sum
    
    def ignored_filter_object_info(self):
        filtered_object_info = {}
        for object_type in [0, 1, 2, 4]:
            obj_map_indices = np.where(self.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.id_map[i, j]
                filtered_object_info[id] = self.object_info[id]
        for obj in self.object_info:
            #TODO: test???
            if self.object_info[obj]['type'] == 3:
                filtered_object_info[obj] = self.object_info[obj]
        return filtered_object_info

    def explore(self, random_prob = 0.1, run_away = False, main_agent_pos = None):
        r"""
            return a unexplored place
            10%: random
            90%: the nearest unexplored place by A*
        """
        if random.random() < random_prob:
            random_flag = True
        else:
            random_flag = False

        try_step = 0

        if not run_away:
            curr_min_distance = 1e11
            curr_x, curr_z = None, None
            while try_step < 10000:
                # try 10000 steps. After 100 steps, if we find a place, we stop.
                try_step += 1
                explore_target = np.where(self.known_map == 0)
                idx = random.randint(0, explore_target[0].shape[0] - 1)
                i, j = explore_target[0][idx], explore_target[1][idx]
                if i < 3 or i > self.map_size[0] - 3 or j < 3 or j > self.map_size[1] - 3:
                    continue
                # a small circle around the pos is zero
                if self.occupancy_map[i, j] == 0 and self.sum_circle(self.occupancy_map + self.known_map, i, j, CIRCLE_RADIUS) == 0:
                    x, z = self.map2pos(i, j)
                    path, obstacle_distance = self.find_shortest_path(self.position, (x, z))
                    if obstacle_distance > EXPLORE_MAX_COST:
                        continue
                    distance = len(path) * 2 + obstacle_distance
                    l2_distance = self.l2_distance(self.position, (x, z))
                    if distance < curr_min_distance and l2_distance > SURROUNDING_DISTANCE:
                        curr_min_distance = distance
                        curr_x, curr_z = x, z
                        if try_step > 100:
                            break
                    if random_flag and curr_x is not None:
                        return curr_x, curr_z
        else:
            points = []
            while try_step < 10000 and len(points) < 500:
                try_step += 1
                explore_target = np.where(np.ones_like(self.known_map, dtype=bool))
                idx = random.randint(0, explore_target[0].shape[0] - 1)
                i, j = explore_target[0][idx], explore_target[1][idx]
                # a small circle around the pos is zero
                if i < 6 or i > self.map_size[0] - 6 or j < 6 or j > self.map_size[1] - 6:
                    continue

                if self.occupancy_map[i, j] == 0 and self.sum_circle(self.occupancy_map, i, j, 1) == 0:
                    x, z = self.map2pos(i, j)
                    distance = self.l2_distance(main_agent_pos, (x, z))
                    l2_distance = self.l2_distance(self.position, (x, z))
                    angle = self.get_angle(forward = np.array([x, 0, z]) - np.array(self.position), origin = np.array(self.position), position = np.array(main_agent_pos))
                    if l2_distance > 5 and (angle > 60 or angle < -60 or (distance > 10 and (angle > 30 or angle < -30))):
                        points.append((distance, x, z))

            points = sorted(points, key=lambda x: x[0])
            
            idx = min(int(len(points) * 3 / 4), len(points) - 1)
            curr_x, curr_z = points[idx][1], points[idx][2]

        while curr_x is None:
            # if we can't find a place, we randomly choose a place
            explore_target = np.where(np.ones_like(self.known_map, dtype=bool))
            idx = random.randint(0, explore_target[0].shape[0] - 1)
            i, j = explore_target[0][idx], explore_target[1][idx]
            x, z = self.map2pos(i, j)
            # a small circle around the pos is zero
            path, obstacle_distance = self.find_shortest_path(self.position, (x, z))
            if obstacle_distance > EXPLORE_MAX_COST:
                continue
            distance = len(path) * 2 + obstacle_distance
            l2_distance = self.l2_distance(self.position, (x, z))
            print(l2_distance)
            if distance < curr_min_distance and l2_distance > SURROUNDING_DISTANCE:
                curr_min_distance = distance
                curr_x, curr_z = x, z

        assert curr_x is not None
        return curr_x, curr_z
    
    def merge_action_and_location(self, action_type) -> dict:
        '''
        action:
           type: 0: move forward, 1: turn left, 2: turn right, 3: pick up, 4: put in, 5: put on
           arms: 0: left, 1: right
           object: object_id
        '''
        #TODO: leave for test
        oppo_agent_pos = self.oppo_pos
        closest_distance = 1e11
        target_object = None
        if action_type == 3: #pick up
            for obj in self.object_info:
                if self.object_info[obj]['type'] != 0 and self.object_info[obj]['type'] != 1: continue
                distance = self.l2_distance(oppo_agent_pos, self.object_info[obj]['position'])
                if distance < closest_distance:
                    closest_distance = distance
                    target_object = obj
            text_description = f"pick up object {self.object_info[target_object]['name']}"
        elif action_type == 4: #put in
            text_description = f"put the object in the container"
        elif action_type == 5: #put on
            for obj in self.object_info:
                if self.object_info[obj]['type'] != 2: continue
                distance = self.l2_distance(oppo_agent_pos, self.object_info[obj]['position'])
                if distance < closest_distance:
                    closest_distance = distance
                    target_object = obj
            
            if target_object is not None:
                text_description = f"put the object on {self.object_info[target_object]['name']}"
            else:
                text_description = f"put the object on unknown place"
        else:
            text_description = "moving"
        text_description += f" at frame {self.obs['current_frames']}"
        return text_description

    def nearest_object_id(self, object_name, object_type):
        r"""
            return the nearest object id of the given object name and object type
        """
        object_id = None
        min_distance = 1e11
        for obj in self.object_info:
            distance = self.l2_distance(self.position, self.object_info[obj]['position'])
            if distance < min_distance and self.object_info[obj]['name'] == object_name and self.object_info[obj]['type'] == object_type:
                min_distance = distance
                object_id = obj
        return object_id

    def dist_to_goalplace(self):
        r"""
            return the distance to the goal place
        """
        goal_places = np.where(self.known_map == 3)
        min_distance = 1e11
        for i in range(goal_places[0].shape[0]):
            distance = self.l2_distance(self.position, self.map2pos(goal_places[0][i], goal_places[1][i]))
            if distance < min_distance:
                min_distance = distance
        return min_distance