from utils import utils_environment as utils
import sys
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{curr_dir}/../../virtualhome/simulation/')
print(f'{curr_dir}/../../virtualhome/simulation/')

from environment.unity_environment import UnityEnvironment as BaseUnityEnvironment
from evolving_graph import utils as utils_env
import pdb
import numpy as np
import copy
import ipdb
from functools import partial
import re


class UnityEnvironment(BaseUnityEnvironment):

	def __init__(self,
				 num_agents=2,
				 max_episode_length=200,
				 env_task_set=None,
				 observation_types=None,
				 agent_goals=None,
				 use_editor=False,
				 base_port=np.random.randint(11000, 13000),  # sjmchangeit from 8080 to 12345
				 port_id=0,
				 executable_args={},
				 recording_options={'recording': False,
									'output_folder': None,
									'file_name_prefix': None,
									'cameras': 'PERSON_FROM_BACK',
									'modality': 'normal'},
				 data_collection = False,
				 data_collection_dir = None,
				 save_image = False,
				 gt_seg = True,
				 seed=123):

		if agent_goals is not None:
			self.agent_goals = agent_goals
		else:
			self.agent_goals = ['full' for _ in range(num_agents)]

		self.task_goal, self.goal_spec = {0: {}, 1: {}}, {0: {}, 1: {}}
		self.env_task_set = env_task_set
		super(UnityEnvironment, self).__init__(
			num_agents=num_agents,
			max_episode_length=max_episode_length,
			observation_types=observation_types,
			use_editor=use_editor,
			base_port=base_port,
			port_id=port_id,
			executable_args=executable_args,
			recording_options=recording_options,
			seed=seed
		)
		self.full_graph = None
		self.num_camera_per_agent = 9
		self.CAMERA_NUM = 8 
		self.default_image_width = 512
		if self.observation_types[0] == 'full_image':
			self.default_image_width = 1024		
		self.default_image_height = 256
		self.message_said = [None for _ in range(num_agents)]
		self.gt_seg = gt_seg
		self.save_image = save_image

		#For collecting data
		self.data_collection = data_collection
		self.data_collection_dir = data_collection_dir
		self.global_episode_id = 0
		self.container_classes = [
			'bathroomcabinet',
			'kitchencabinet',
			'cabinet',
			'fridge',
			'stove',
			# 'kitchencounterdrawer',
			# 'coffeepot',
			'dishwasher',
			'microwave']
		self.detection_all_object = ['plate', 'coffeetable', 'poundcake', 'wine', 'juice', 'pudding', 'apple', 'pancake', 'cutleryfork', 'kitchentable', 'cupcake', 'coffeepot'] + self.container_classes + ['character']
		self.detection_name_id_map = {name:  (i + 1) * 10 for i, name in enumerate(self.detection_all_object)}
		self.location = None
		self.keep_move_steps = None
 
	def reward(self):
		reward = 0.
		done = True
		satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
		for key, value in satisfied.items():
			preds_needed, mandatory, reward_per_pred = self.goal_spec[0][key]
			# How many predicates achieved
			value_pred = min(len(value), preds_needed)
			reward += value_pred * reward_per_pred

			if mandatory and unsatisfied[key] > 0:
				done = False

		self.prev_reward = reward
		return reward, done, {'satisfied_goals': satisfied}

	def get_action_space(self):
		dict_action_space = {}
		for agent_id in range(self.num_agents):
			if 'image' in self.observation_types[agent_id]:
				return None
			#In vision, no RL is applied
			if self.observation_types[agent_id] not in ['partial', 'full']:
				raise NotImplementedError
			# TODO:  Now the action space is base on graph
			else:
				# Even if you can see all the graph, you can only interact with visible objects
				obs_type = 'partial'
			visible_graph = self.get_observation(agent_id, obs_type)
			dict_action_space[agent_id] = [node['id'] for node in visible_graph['nodes']]
		return dict_action_space

	def check_subgoal(self, subgoal):
		# Check if the subgoal is satisfied
		# If the subgoal is satisfied, return True
		# If the subgoal is not satisfied, return False
		if subgoal is None: return False
		if type(subgoal) == list: 
			if len(subgoal) == 0: return False
			else: subgoal = subgoal[0]
		id_to_grab = int(subgoal.split('_')[1])
		class_to_grab = self.id_to_name[id_to_grab]
		satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
		for x in satisfied.keys():
			for t in satisfied[x]:
				if t is None: continue
				if str(id_to_grab) in t:
					print(subgoal, 'is satisfied')
					return True
		for x in unsatisfied.keys():
			if class_to_grab in x and unsatisfied[x] == 0:
				print(subgoal, 'is satisfied')
				return True
		return False
	
	def get_goal(self, task_spec, agent_goal):
		if agent_goal == 'full':
			pred = [x for x, y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']]
			# object_grab = [pr.split('_')[1] for pr in pred]
			# predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
			res_dict = {goal_k: [goal_c, True, 2] for goal_k, goal_c in task_spec.items()}
			# res_dict.update(predicates_grab)
			return res_dict
		elif agent_goal == 'grab':
			candidates = [x.split('_')[1] for x, y in task_spec.items() if
						  y > 0 and x.split('_')[0] in ['on', 'inside']]
			object_grab = self.rnd.choice(candidates)
			# print('GOAL', candidates, object_grab)
			return {'holds_' + object_grab + '_' + '1': [1, True, 10],
					'close_' + object_grab + '_' + '1': [1, False, 0.1]}
		elif agent_goal == 'put':
			pred = self.rnd.choice([x for x, y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']])
			object_grab = pred.split('_')[1]
			return {
				pred: [1, True, 60],
				'holds_' + object_grab + '_' + '1': [1, False, 2],
				'close_' + object_grab + '_' + '1': [1, False, 0.05]

			}
		elif agent_goal == 'LLM':  # todo: Hongxin added
			'''
			Hongxin changed task_goal from "on_cupcake_268" to "on_cupcake_<coffeetable> (268)"
			'''
			goal_class = {}
			for predicate in self.goal_class.keys():
				rel, obj1, obj2 = predicate.split('_')
				goal_class[f"{rel}_{obj1}"] = obj2
			new_task_goal = {}
			for predicate, count in task_spec.items():
				if count == 0:
					continue
				rel, obj1, obj2 = predicate.split('_')
				obj2_name = goal_class[f"{rel}_{obj1}"]
				new_predicate = predicate.replace(obj2, f"<{obj2_name}> ({obj2})")
				new_task_goal[new_predicate] = count
			# print(new_task_goal)
			res_dict = {goal_k: [goal_c, True, 2] for goal_k, goal_c in new_task_goal.items()}
			# res_dict.update(predicates_grab)
			return res_dict
		else:
			raise NotImplementedError

	@property
	def all_relative_name(self) -> list:
		return self.all_containers_name + self.all_goal_objects_name + ['character']

	@property
	def all_relative_id(self) -> list:
		return [node['id'] for node in self.full_graph['nodes'] if node['class_name'] in self.all_relative_name]

	@property
	def all_detection_id(self) -> list:
		return [node['id'] for node in self.full_graph['nodes'] if node['class_name'] in self.detection_all_object]

	@property
	def all_containers_name(self) -> list:
		r'''
		 get all containers in the scene, exclude rooms and containers with no objects inside.
		'''
		'''
		id2node = {node['id']: node for node in self.full_graph['nodes']}
		room_name = [node['class_name'] for node in self.full_graph['nodes'] if node['category'] == 'Rooms']
		all_container = list(set([id2node[link['to_id']]['class_name'] for link in self.full_graph['edges'] if
								  link['relation_type'] == 'INSIDE']))
		all_container = [x for x in all_container if x not in room_name]
		'''
		container_classes = [
			'bathroomcabinet',
			'kitchencabinet',
			'cabinet',
			'fridge',
			'stove',
			# 'coffeepot',
			'dishwasher',
			'microwave']
		return container_classes

	@property
	def all_goal_objects_name(self) -> list:
		r'''
		 get all objects that related to goal.
		ZHX: update to adapt to new goal_spec of LLM
		'''
		goal_objects = []
		id2node = {node['id']: node for node in self.full_graph['nodes']}
		for predicate in self.goal_spec[0]:
			elements = predicate.split('_')
			for x in elements[1:]:
				if x.isdigit():
					goal_objects += [id2node[int(x)]['class_name']]
				elif '(' in x:
					y = x.split('(')[1].split(')')[0]
					if y.isdigit():
						goal_objects += [id2node[int(y)]['class_name']]
				else:
					goal_objects += [x]
		goal_obj = list(set(goal_objects))
		# if ('character' not in goal_obj):
		# 	goal_obj += ['character']
		return goal_obj

	@property
	def room_info(self):
		r'''
		 get room info in the scene.
		'''
		return [node for node in self.full_graph['nodes'] if node['id'] in self.all_room_and_character_id]

	@property
	def all_room_name(self) -> list:
		r'''
		 get all rooms in the scene.
		'''
		# room_name = [node['class_name'] for node in self.full_graph['nodes'] if node['category'] == 'Rooms']
		room_name = ["livingroom", "kitchen", "bedroom", "bathroom"]
		return room_name


	@property
	def all_room_and_character_id(self) -> list:
		r'''
		get all room_and_character_ids in the scene.
		'''
		return [node['id'] for node in self.full_graph['nodes'] if
						  node['class_name'] == 'character' or node['category'] in ['Rooms']]
	
	@property
	def all_room_id(self) -> list:
		r'''
		get all room_and_character_ids in the scene.
		'''
		return [node['id'] for node in self.full_graph['nodes'] if node['category'] in ['Rooms']]

	def filter_graph(self, obs):
		relative_id = self.all_relative_id + self.all_room_id
		new_graph = {
			"edges": [edge for edge in obs['edges'] if
					edge['from_id'] in relative_id and edge['to_id'] in relative_id],
			"nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
		}
		return new_graph

	def reset(self, environment_graph=None, task_id=None):
		# Make sure that characters are out of graph, and ids are ok
		# ipdb.set_trace()
		self.global_episode_id += 1 # For collecting data
		if task_id is None:
			task_id = self.rnd.choice(list(range(len(self.env_task_set))))
		env_task = self.env_task_set[task_id]

		self.task_id = env_task['task_id']
		self.init_graph = copy.deepcopy(env_task['init_graph'])
		self.init_rooms = env_task['init_rooms']
		self.task_goal = env_task['task_goal']
		print('task_goal: ', self.task_goal)
		if 'total_goal' in env_task.keys():
			print('total_goal: ', env_task['total_goal'])
		self.goal_class = env_task['goal_class']
		self.task_name = env_task['task_name']
		self.cache_id_map = dict()
		self.cache_data_map = dict() # For collecting data

		old_env_id = self.env_id
		self.env_id = env_task['env_id']
		print("Resetting... Envid: {}. Taskid: {}. Index: {}".format(self.env_id, self.task_id, task_id))

		# TODO: in the future we may want different goals
		self.goal_spec = {agent_id: self.get_goal(self.task_goal[agent_id], self.agent_goals[agent_id])
						  for agent_id in range(self.num_agents)}

		if False:  # old_env_id == self.env_id:
			print("Fast reset")
			self.comm.fast_reset()
		else:
			self.comm.reset(self.env_id)

		s, g = self.comm.environment_graph()
		edge_ids = set([edge['to_id'] for edge in g['edges']] + [edge['from_id'] for edge in g['edges']])
		node_ids = set([node['id'] for node in g['nodes']])
		if len(edge_ids - node_ids) > 0:
			pdb.set_trace()

		if self.env_id not in self.max_ids.keys():
			max_id = max([node['id'] for node in g['nodes']])
			self.max_ids[self.env_id] = max_id

		max_id = self.max_ids[self.env_id]

		if environment_graph is not None:
			updated_graph = environment_graph
			s, g = self.comm.environment_graph()
			updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
			success, m = self.comm.expand_scene(updated_graph)
		else:
			updated_graph = self.init_graph
			s, g = self.comm.environment_graph()
			updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
			success, m = self.comm.expand_scene(updated_graph)

		if not success:
			ipdb.set_trace()
			print("Error expanding scene")
			ipdb.set_trace()
			return None

		self.offset_cameras = self.comm.camera_count()[1]
		if self.init_rooms[0] not in ['kitchen', 'bedroom', 'livingroom', 'bathroom']:
			rooms = self.rnd.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
		else:
			rooms = list(self.init_rooms)

		self.num_static_cameras = self.comm.camera_count()[1]
		s, g = self.comm.add_character_camera(position=[0, 1.8, 0.15], rotation=[30,0,90], name="up_camera")
		#field_view=90, does not work, since the unity env do not support it
		for i in range(self.num_agents):
			if i in self.agent_info:
				self.comm.add_character(self.agent_info[i], initial_room=rooms[i])
			else:
				self.comm.add_character()
		_, self.init_unity_graph = self.comm.environment_graph()

		self.changed_graph = True
		graph = self.get_graph()
		self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
		self.id2node = {node['id']: node for node in graph['nodes']}

		curr_graph = self.get_graph()
		curr_graph = utils.inside_not_trans(curr_graph)
		self.full_graph = curr_graph
		self.id_to_name = {node['id']: node['class_name'] for node in curr_graph['nodes']}
		self.message_said = [None for _ in range(self.num_agents)]

		self.location = [[] for _ in range(self.num_agents)]
		self.last_script = [None for _ in range(self.num_agents)]
		self.keep_move_steps = [0 for _ in range(self.num_agents)]
		obs = self.get_observations()
		self.steps = 0
		self.prev_reward = 0.
		return obs

	def handle_script(script_list):
		pass
	
	def range_distance(self, pos_list):
		max_dis = 0
		for i in range(len(pos_list)):
			for j in range(i+1, len(pos_list)):
				dis = np.linalg.norm(np.array(pos_list[i]) - np.array(pos_list[j]))
				if dis > max_dis:
					max_dis = dis
		return max_dis

	def step(self, action_dict):
		if self.steps > 245:
			print("Warning: too many steps")
		K = 500
		actions = [utils.get_action_name(x) for x in action_dict.values()]
		action_dict_verbose = copy.deepcopy(action_dict)
		action_dict_tobe_changed = copy.deepcopy(action_dict)
		saying = [None for _ in range(len(actions))]
		say = False
		for i, action in enumerate(actions):
			random_flag = False
			if action == 'send_message':
				say = True
				saying[i] = utils.get_message_name(action_dict_tobe_changed[i])
				if len(saying[i]) > K:
					saying[i] = saying[i][:K]
					print("Message too long, truncating to {}".format(K))
				action_dict_tobe_changed[i] = None
			# Add restrictions on 'open' action, since open a container twice will cause an env error
			if action == 'open':
				goal_obj = int(re.findall(r'\d+', action_dict_tobe_changed[i])[0])
				if 'OPEN' in self.get_states(goal_obj):
					action_dict_tobe_changed[i].replace('open', 'walktowards')
					assert False, "Error: Object already open"
			# Add randomness on 'walktowards' action, since the env's walktowards action might stuck
			if action == 'walktowards' and self.keep_move_steps[i] > 4:
				if self.range_distance(self.location[i][max(0, self.steps - 4) : self.steps + 1]) < 0.5:
					random_place = self.rooms[np.random.randint(0, len(self.rooms) - 1)]
					action_dict_tobe_changed[i] = '[walktowards] <{}> ({})'.format(random_place[0], random_place[1])
					print("Randomly changing walktowards action since the agent might get stuck")
					random_flag = True
			self.keep_move_steps[i] = self.keep_move_steps[i] + 1 if action == 'walktowards' and random_flag == False else 0

		script_list = utils.convert_action(action_dict_tobe_changed)
		script_list_verbose = utils.convert_action(action_dict_verbose)
		failed_execution = False
		print(f"Step {self.steps}, Executing script: {script_list_verbose}")

		if len(script_list[0]) > 0:
			if self.recording_options['recording']:
				assert False, "Recording not supported"
				success, message = self.comm.render_script(script_list,
														   recording=True,
														   # gen_vid=False,
														   skip_animation=False,
														   camera_mode=self.recording_options['cameras'],
														   file_name_prefix='task_{}'.format(self.task_id),
														   image_synthesis=self.recording_optios['modality'])
			else:
				individual_script = script_list[0].split('|')
				for i in range(len(individual_script)):
					success, message = self.comm.render_script([individual_script[i]],
														   recording=False,
														   image_synthesis=[],
														   # gen_vid=False,
														   skip_animation=True)
					if not success:
						print("NO SUCCESS")
						print(message, script_list)
						failed_execution = True
					else:
						self.changed_graph = True

		# Obtain reward
		reward, done, info = self.reward()

		graph = self.get_graph()
		self.steps += 1

#		obs = self.get_observations()
		obs = None
		# already get it in env.get_observations()

		info['finished'] = done
		info['graph'] = graph
		info['failed_exec'] = failed_execution
		satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
		info['progress'] = {'satisfied': satisfied, 'unsatisfied': unsatisfied}
		
		if self.steps == self.max_episode_length:
			done = True
		messages = saying
		self.message_said = saying		

		if len(script_list[0]) == 0 and say == False and self.data_collection == True:
			#stacked, recollect data	
			print("Stacked")
			done = True

		return obs, reward, done, info, messages

	def get_observations(self):
		curr_graph = self.get_graph()
		curr_graph = utils.inside_not_trans(curr_graph)
		self.full_graph = curr_graph
		self.dict_graph = {node['id']: node for node in curr_graph['nodes']}

		self.id_map = []
		dict_observations = {}
		for agent_id in range(self.num_agents):
			obs_type = self.observation_types[agent_id]
			if ('image' in obs_type or self.data_collection):
				image_width, image_height = self.default_image_width, self.default_image_height
				camera_ids = [self.num_static_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
				s, seg_inst = self.comm.camera_image(camera_ids, mode='seg_inst', image_width=image_width,
													 image_height=image_height)
				s, inst_colors = self.comm.instance_colors()

				def find_id_by_colors(rgb):
					if rgb in self.cache_id_map.keys():
						return self.cache_id_map[rgb]
					ans_id = []
					for k, v in inst_colors.items():
						if abs(rgb[0] - v[2] * 255) + abs(rgb[1] - v[1] * 255) + abs(rgb[2] - v[0] * 255) < 10:
							if int(k) in self.all_relative_id:
								ans_id.append(int(k))
					ans = ans_id[0] if len(ans_id) == 1 else -1
					assert(len(ans_id) <= 1)
					self.cache_id_map[rgb] = ans
					return ans

				ids = np.empty((len(camera_ids), image_height, image_width))
				inside_info = {}
				for t in range(len(camera_ids)):
					for i in range(image_height):
						for j in range(image_width):
							ids[t, i, j] = find_id_by_colors(tuple(seg_inst[t][i, j]))
							if (ids[t, i, j] != -1):
								if (ids[t, i, j] not in inside_info.keys()):
									inside_info[ids[t, i, j]] = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == ids[t, i, j] and edge['relation_type'] == 'INSIDE'][0]
								if (self.dict_graph[inside_info[ids[t, i, j]]]['category'] not in ['Rooms'] and 'CLOSED' in self.dict_graph[inside_info[ids[t, i, j]]]['states']): 
									ids[t, i, j] = -1
				self.id_map.append(ids)
				colorids = np.stack(((ids % 10) * 10, ((ids // 10) % 10) * 10, ((ids // 100) % 10) * 10), axis=3)
				if self.save_image:
					import cv2
					for i in range(len(camera_ids)):
						rot_seg_inst = np.rot90(seg_inst[i], axes = (0, 1))
						rot_colorids = np.rot90(colorids[i], axes = (0, 1))
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_seg.png"), rot_seg_inst)
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_seg_reference.png"), rot_colorids)
			dict_observations[agent_id] = self.get_observation(agent_id, obs_type)
			self.location[agent_id].append(dict_observations[agent_id]['location'])
			if self.data_collection:
				def find_data_by_colors(rgb):
					if rgb in self.cache_data_map.keys():
						return self.cache_data_map[rgb]
					ans_id = []
					for k, v in inst_colors.items():
						if abs(rgb[0] - v[2] * 255) + abs(rgb[1] - v[1] * 255) + abs(rgb[2] - v[0] * 255) < 10:
							if (int(k) in self.all_detection_id):
								ans_id.append(int(k))
					ans = ans_id[0] if len(ans_id) == 1 else -1
					assert(len(ans_id) <= 1)
					self.cache_data_map[rgb] = ans
					return ans
				inside_info = {}
				ids_class = np.empty((len(camera_ids), image_height, image_width))
				ids_instance = np.empty((len(camera_ids), image_height, image_width))
				for t in range(len(camera_ids)):
					for i in range(image_height):
						for j in range(image_width):
							ids_class[t, i, j] = find_data_by_colors(tuple(seg_inst[t][i, j]))
							if ids_class[t, i, j] != -1:
								if ids_class[t, i, j] not in inside_info.keys():
									inside_info[ids_class[t, i, j]] = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == ids_class[t, i, j] and edge['relation_type'] == 'INSIDE'][0]
								if self.dict_graph[inside_info[ids_class[t, i, j]]]['category'] not in ['Rooms'] and 'CLOSED' in self.dict_graph[inside_info[ids_class[t, i, j]]]['states']:
									ids[t, i, j] = -1
							if ids_class[t, i, j] != -1:
								ids_instance[t, i, j] = ids_class[t, i, j]
								ids_class[t, i, j] = self.detection_name_id_map[self.id_to_name[ids_class[t, i, j]]]
							else:
								ids_instance[t, i, j] = -1
				color_id_instance = np.stack(((ids_instance % 10) * 10, ((ids_instance // 10) % 10) * 10, ((ids_instance // 100) % 10) * 10), axis=3)
				if 'bgr' not in dict_observations[agent_id].keys():
					s, dict_observations[agent_id]['bgr'] = self.comm.camera_image(camera_ids, mode='normal', image_width=image_width,
													 image_height=image_height)
				if os.path.exists(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/') == False:
					os.mkdir(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/')
				import cv2
				for t in range(len(camera_ids)):
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_ids_instance.png', ids_instance[t])
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_color_ids_instance.png', color_id_instance[t])
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_ids_class.png', ids_class[t])
					cv2.imwrite(self.data_collection_dir + str(self.global_episode_id) + '_' + str(self.env_id) + '/' + str(self.steps) + '_' + str(agent_id) + '_rgb.png', dict_observations[agent_id]['bgr'][t])
		return dict_observations

	def get_agent_location(self, agent_id):
		graph = self.full_graph
		agent_node = \
			[node for node in graph['nodes'] if node['category'] == 'Characters' and node['id'] == agent_id + 1][0]
		return agent_node['obj_transform']['position']

	def bbox_3d_to_id(self, bbox):
		r'''
			Given a bounding box, return the id of the object in the scene
			api for agent to get the id of the object

			return: id and name of the object, if the main object is not relative object, return (-1, None).
		'''
		# deprecate now
		raise NotImplementedError

	def bbox_2d_to_id(self, bbox, agent_id, frame_id = 0, mask = None, threshold = 0.3):
		r'''
			Given a bounding box, return the id of the object in the scene
			api for agent to get the id of the object
			bbox: [x_min, y_min, x_max, y_max]

			return: id and name of the object, if the main object is not relative object, return (-1, None).
		'''
		id_array = copy.deepcopy(self.id_map[agent_id][frame_id, int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])])
		if (mask is not None):
			mask = mask[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
			total_pixels = np.sum(mask)
			id_array[mask == False] = -1
		else:
			total_pixels = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
		id_array = id_array.reshape(-1)
		id_array = id_array[id_array != -1]
		if len(id_array) == 0:
			return -1, None
		count = np.bincount(id_array.astype(int))
		d = np.argmax(count)
		if (count[d] < total_pixels * threshold):
			return -1, None
		if (d in self.all_relative_id):
			return d, self.id_to_name[d]
		else:
			return -1, None
		
	def id_to_inside_room(self, id):
		r'''
			Given an id, return the id of the room it is inside.
			api for agent to get the id of the room it is inside.

			return: id and name of the room, if the main object is not relative object, return (-1, None).
		'''
		while (id not in self.all_room_id):
			id = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == id and edge['relation_type'] == 'INSIDE'][0]
		return id

	def get_properties(self, id):
		r'''
			return 'properties' of the object with id.
		'''
		return [node['properties'] for node in self.full_graph['nodes'] if node['id'] == id][0]
	
	def get_category(self, id):
		r'''
			return 'category' of the object with id.
		'''
		return [node['category'] for node in self.full_graph['nodes'] if node['id'] == id][0]
	
	def get_states(self, id):
		r'''
			return 'states' of the object with id.
		'''
		return [node['states'] for node in self.full_graph['nodes'] if node['id'] == id][0]
	
	def get_class_name(self, id):
		r'''
			return 'states' of the object with id.
		'''
		return [node['class_name'] for node in self.full_graph['nodes'] if node['id'] == id][0]

	def nodes_in_same_room(self, nodes, agent_id = 0):
		same_room_node =  [node for node in nodes if self.id_to_inside_room(node['id']) == self.id_to_inside_room(agent_id)]
		not_inside_node = []
		for node in same_room_node:								
			inside_info = [edge['to_id'] for edge in self.full_graph['edges'] if edge['from_id'] == node['id'] and edge['relation_type'] == 'INSIDE']
			assert (len(inside_info) <= 1)
			if len(inside_info) == 0:
				not_inside_node.append(node)
			elif self.dict_graph[inside_info[0]]['category'] in ['Rooms'] or 'CLOSED' not in self.dict_graph[inside_info[0]]['states']:
				not_inside_node.append(node)
		return not_inside_node

	def agent_env_api(self, agent_id) -> dict:
		# all the api the agent can use in the vision env.
		return {
			'bbox_2d_to_id': partial(self.bbox_2d_to_id, agent_id=agent_id),
			'get_properties': self.get_properties,
			'get_category': self.get_category,
			'get_states': self.get_states,
			'nodes_in_same_room': partial(self.nodes_in_same_room, agent_id = (agent_id + 1)),
			'get_class_name': self.get_class_name,
			'remove_duplicate_graph': self.remove_duplicate_graph,
		}

	def remove_duplicate_graph(self, graph):
		# remove duplicate nodes / edges
		allnode = [node['id'] for node in graph['nodes']]
		simplify_edge = []
		all_edge = []
		for edge in graph['edges']:
			if edge['from_id'] in allnode and (edge['to_id'] in allnode or self.id2node[edge['to_id']]['category'] == 'Rooms'):
				if (edge['from_id'], edge['to_id'], edge['relation_type']) not in all_edge:
					simplify_edge.append(edge)
					all_edge.append((edge['from_id'], edge['to_id'], edge['relation_type']))
		return {
			'nodes': graph['nodes'],
			'edges': simplify_edge,
		}

	def clean_object_relationship(self, graph):
		return {
			'nodes': [],
			'edges': [edge for edge in graph['edges'] if (self.id2node[edge['from_id']]['class_name'] == 'character' or self.id2node[edge['to_id']]['class_name'] == 'character') and edge['relation_type'] not in ['ON', 'BETWEEN', 'INSIDE']],
		}
	def get_observation(self, agent_id, obs_type):
		agent_location = self.get_agent_location(agent_id)

		if obs_type == 'partial':
			# agent 0 has id (0 + 1)
			obs = utils_env.get_visible_nodes(self.full_graph, agent_id=(agent_id + 1))
			if self.save_image:
				assert (
						self.num_static_cameras is not None and self.num_camera_per_agent is not None and self.CAMERA_NUM is not None)
				camera_ids = [self.num_static_cameras-1]
				# camera_ids = [self.num_static_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
                # camera_ids = [self.num_static_cameras - 1] has a top-down view
				image_width, image_height = self.default_image_width, self.default_image_height
				# All the types are "normal", "seg_inst", "seg_class", "depth", "flow", "albedo", "illumination", "surf_normals"
				s, bgr_images = self.comm.camera_image(camera_ids, mode='normal', image_width=image_width, image_height=image_height)
				s, depth_images = self.comm.camera_image(camera_ids, mode='depth', image_width=image_width, image_height=image_height)
				s, camera_info = self.comm.camera_data(camera_ids)
				if self.save_image:
					import cv2
					for i in range(len(camera_ids)):
						rot_bgr = np.rot90(bgr_images[i], axes = (0, 1))
						rot_depth = np.rot90(depth_images[i], axes = (0, 1))
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_img.png"), rot_bgr)
						cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_depth.png"), 50 / rot_depth) # for visualization
			return {'messages': self.message_said,**obs, 'location': agent_location}

		elif obs_type == 'full':
			return self.full_graph

		elif obs_type == 'visible':
			# Only objects in the field of viedw of the agent
			raise NotImplementedError

		elif 'image' in obs_type:
			#we have 'full_image' and 'normal_image'
			assert (self.num_static_cameras != None and self.num_camera_per_agent != None and self.CAMERA_NUM != None)
			camera_ids = [self.num_static_cameras + agent_id * self.num_camera_per_agent + self.CAMERA_NUM]
			image_width, image_height = self.default_image_width, self.default_image_height
			# All the types are "normal", "seg_inst", "seg_class", "depth", "flow", "albedo", "illumination", "surf_normals"
			s, bgr_images = self.comm.camera_image(camera_ids, mode='normal', image_width=image_width, image_height=image_height)
			s, depth_images = self.comm.camera_image(camera_ids, mode='depth', image_width=image_width, image_height=image_height)
			s, camera_info = self.comm.camera_data(camera_ids)
			if self.save_image:
				import cv2
				for i in range(len(camera_ids)):
					rot_bgr = np.rot90(bgr_images[i], axes = (0, 1))
					rot_depth = np.rot90(depth_images[i], axes = (0, 1))
					cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_img.png"), rot_bgr)
					cv2.imwrite(os.path.join('saved_images', f"{agent_id}_{self.steps:03}_depth.png"), 50 / rot_depth) # for visualization
			if not s:
				pdb.set_trace()
			if not self.gt_seg:
				return {
				'bgr': bgr_images,  # 1 * w * h * 3
				'depth': depth_images,  # 1 * w * h * 3
				'camera_info': camera_info,
				'room_info': self.room_info,
				'location': agent_location,  # [x, y, z]
				'messages': self.message_said,
				'current_room': self.id_to_inside_room(agent_id + 1),
				**self.agent_env_api(agent_id),
				**self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
						agent_id + 1))),
				}
			else: return {
				'bgr': bgr_images,  # 1 * w * h * 3
				'depth': depth_images,  # 1 * w * h * 3
				'camera_info': camera_info,
				'seg_info': self.id_map[agent_id],
				'room_info': self.room_info,
				'messages': self.message_said,
				'location': agent_location,  # [x, y, z]
				'current_room': self.id_to_inside_room(agent_id + 1),
				**self.agent_env_api(agent_id),
				**self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
						agent_id + 1))),
			}
		else:
			raise NotImplementedError

		return updated_graph