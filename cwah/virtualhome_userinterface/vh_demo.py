#!flask/bin/python
#import cStringIO
import numpy as np
import torch
import time
import json
import ipdb
import sys, os
from collections import namedtuple
import copy

sys.path.append('..')
sys.path.append(os.path.join('..', '..', 'virtualhome', 'simulation'))
from tools.plotting_code_planner import plot_graph_2d_v2 as plot_graph_2d

# import plotly.io
import agents
import utils.utils_environment as utils_environment
import argparse
import pickle as pkl
from flask import Flask, render_template, request, redirect, Response, send_file
# from virtualhome.simulation.unity_simulator import comm_unity
from envs.unity_environment import UnityEnvironment
import vh_tools
import random, json
import cv2
from datetime import datetime
import time
from PIL import Image
import io
import base64

import matplotlib.pyplot as plt

app = Flask(__name__)


image_top = None
# comm = None
env = None
lc = None
instance_colors = None
current_image = None
images = None
prev_images = None
graph = None
id2node = None
aspect_ratio = 9./16.
bad_class_name_ids = []
curr_task = None
last_completed = {}
previous_belief = []
previous_belief_action = []
extra_agent = None
next_helper_action = None
save_pkl_file = "../dataset/test_env_set_help.pik"

# parameters for record graph
time_start = None
record_graph_flag = True # False
#vis_graph_flag = True
graph_save_dir = None

record_step_counter = 1
dialogue_history = []

# Contains a mapping so that objects have a smaller id. One unique pr object+instance
# instead of instance. glass.234, glass.267 bcecome glass.1 glass.2
# class2numobj = {}

task_index = -1 # for indexing task_id in task_group
task_index_shuffle = []
last_instr_main = None

current_goal = {}

parser = argparse.ArgumentParser(description='Collection data simulator.')
parser.add_argument("--deployment", type=str, choices=["local", "remote"], default="remote")
parser.add_argument("--portflask", type=int, default=5005)
parser.add_argument("--executable_file", type=str, help="The location of the executable name")
parser.add_argument("--portvh", type=int, default=8080)
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument('--showmodal', action='store_true')
parser.add_argument("--task_group", type=int, nargs="+", default=[0], help='usage: --task_group 0')
parser.add_argument("--trial", type=int)
parser.add_argument("--exp_name", type=str, default="debugtest")
parser.add_argument("--extra_agent", type=str, choices = ["MCTS_comm", "LLM_comm", "LLM", "none"], default='none')
parser.add_argument("--dataset", type=str, default="../dataset/test_env_set_help.pik")
parser.add_argument('--communication', action='store_true', default=False,
						help='enabling communication')
# LLM parameters
parser.add_argument('--source', default='openai',
	choices=['huggingface', 'openai', 'debug'],
	help='openai API or load huggingface models')
parser.add_argument('--lm_id', default='gpt-3.5-turbo',
					help='name for openai engine or huggingface model name/path')
parser.add_argument('--prompt_template_path', default='../LLM/prompt_nocom.csv',
					help='path to prompt template file')
parser.add_argument("--t", default=0.7, type=float)
parser.add_argument("--top_p", default=1.0, type=float)
parser.add_argument("--max_tokens", default=256, type=int)
parser.add_argument("--n", default=1, type=int)
parser.add_argument("--logprobs", default=1, type=int)
parser.add_argument("--cot", action='store_true', help="use chain-of-thought prompt")
parser.add_argument("--debug", action='store_true')
parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")

args = parser.parse_args()

# def get_pred_name(pred_name, id2node):
#	 sp = pred_name.split('_')
#	 return '_'.join([sp[0], sp[1], id2node[int(sp[2])]])


# def add_goal_class(current_task):
#	 task_goal = current_task['task_goal'][0]
#	 id2class = {node['id']: node['class_name'] for node in current_task['init_graph']['nodes']}
#	 goal_class = {get_pred_name(pred, id2class): count for pred, count in task_goal.items()}
#	 current_task['goal_class'] = goal_class

def convert_image(img_array):
	#cv2.imwrite('current.png', img_array.astype('uint8'))
	img = Image.fromarray(img_array.astype('uint8'))
	#file_like = cStringIO.cStringIO(img)
	file_object = io.BytesIO()
	img.save(file_object, 'JPEG')
	img_str = base64.b64encode(file_object.getvalue())
	return img_str
	#img.save(img, 'current.png')
	#file_object.seek(0)
	#return file_object

def send_command(command):
	print('send command:', command)
	global graph, id2node, record_step_counter, time_start, last_completed, next_helper_action, curr_task
	global task_index, current_goal, extra_agent, last_instr_main, id2node
	global dialogue_history, previous_belief, previous_belief_action
	executed_instruction = False

	info = {}
	script = {}
	g = env.get_graph()
	graph = g
	savegraph = graph
	# TODO: make this more general
	id2node = {node['id']: node for node in graph['nodes']}
	ai_send_message = None
	agent_info = None

	if task_index >= len(args.task_group):
		return [], {'all_done': True}
	if command['instruction'] == 'reset':
		# executed_instruction = True
		if 'data_form' in command:
			with open(os.path.join(graph_save_dir, 'final_form.json'), 'w') as f:
				f.write(json.dumps(command['data_form']))
		print('reset', command['scene_id'])
		images, info = reset(command['scene_id'])
		# ipdb.set_trace()
		if info['all_done']:
			return [], {'all_done': True}


		# return images, info
	else:
		failed_script = False
		instr = command['instruction']
		if instr == 'refresh':
			image_top = 'include_top' in command
			if image_top:
				info['image_top'] = get_top_image()

		else:
			other_info = None
			if 'other_info' in command:
				other_info = command['other_info']

			action = instr
			# breakpoint()
			if action == 'send_message':
				human_message = other_info[0][0] # actual message
				dialogue_history.append('Alice: ' + human_message)
			else:
				if other_info is not None:
					object_name, object_id = other_info[0]
				else:
					object_name, object_id = None, None
			if action == 'wait':
				human_script = None
			else:
				#sjm this can_perform_action is to be changed, but we just place it here.
				if action != 'send_message':
					human_script, message = vh_tools.can_perform_action(action, object_name, object_id, graph, id2node=id2node)
				else:
					human_script, message = f'[send_message] <{human_message}>', {'msg': "Success"}
				if human_script is None:
					failed_script = True

			if not failed_script:
				executed_instruction = True
				prev_instr_main = last_instr_main
				last_instr_main = human_script
				script[0] = human_script

				# command_other = '[grab] <wineglass> (389)'
				# script[0] = '<char1> {}'.format(command_other)
				if extra_agent is not None:
					#sjm bugs: get_helper_action always return None
					goal_spec = env.get_goal(env.task_goal[1], env.agent_goals[1])
					command_other, agent_info = get_helper_action(goal_spec, prev_instr_main, record_step_counter)
					print('command_other=', command_other)
					if utils_environment.get_action_name(command_other) == 'send_message':
						ai_send_message = utils_environment.get_message_name(command_other)
						dialogue_history.append('Bob: ' + ai_send_message)
					script[1] = command_other
				if script is not None:
					env.step(script)
					env.get_observations()#sjm must run, else env info are not updated
					# comm.render_script(script, skip_animation=True, recording=False, image_synthesis=[])

			else:
				print(message, "ERROR")
				info.update({'errormsg': message})

	g = env.get_graph()
	graph = g

	id2node = {node['id']: node for node in graph['nodes']}
	current_room = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'INSIDE'][0]
	rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'].lower() == 'rooms']
	info.update({'rooms': rooms})

	rooms_id = [i[1] for i in rooms]


	# For now always

	# if extra_agent is not None:
	# goal_spec = {}
	#next_helper_action = get_helper_action(graph, goal_spec, last_instr_main, record_step_counter)
	# print("GETTING ACTION")
	obs = env.get_observation(0, 'partial')
	visible_graph = {'edges':obs['edges'], 'nodes':obs['nodes']}

	filtered_graph = env.filter_graph(visible_graph)
	# breakpoint()
	object_action, other_info = vh_tools.graph_info(filtered_graph)
	# object_action, other_info = vh_tools.graph_info(env.get_graph())
	for it, (obj, actions) in enumerate(object_action):
		if object_action[it][0][4] == '1':
			object_action[it] = (object_action[it][0], [action for action in object_action[it][1] if action != 'walktowards'])
	# If the object is close, remove walktowards


	#print(object_action)
	#print(other_info.keys())

	other_info['ai_send_message'] = ai_send_message

	# Information about task name
	other_info['task_name'] = curr_task['task_name']
	# Task preds
	if len(last_completed) == 0:
		#ipdb.set_trace()
		for task_pred, count in curr_task['task_goal'][0].items():
			last_completed[task_pred] = 0
			# total_completed[task_pred] = 0

	first_pred = list(curr_task['task_goal'][0].keys())[0]
	obj_id_pred = int(first_pred.split('_')[-1])
	visible_ids = [v[0] for v in other_info['visible_objects']]

	num_preds_done = 0
	num_preds_needed = 0
	num_preds_done_all = 0

	# Everything is on or inside...
	total_completed = {}
	ids_curr = None
	if first_pred.split('_')[0] == 'on':
		ids_curr = [id2node[edge['from_id']]['class_name'] for edge in g['edges'] if edge['relation_type'] == 'ON' and edge['to_id'] == obj_id_pred]
	elif first_pred.split('_')[0] == 'inside':
		ids_curr = [id2node[edge['from_id']]['class_name'] for edge in g['edges'] if edge['relation_type'] == 'INSIDE' and edge['to_id'] == obj_id_pred]

	grabbed = [id2node[edge['to_id']]['class_name'] for edge in g['edges'] if 'HOLD' in edge['relation_type'] and edge['from_id'] == 1]
	# print([edge for edge in g['edges'] if 'HOLD' in edge['relation_type']])
	if ids_curr is not None:
		for task_pred in curr_task['task_goal'][0].keys():
			if 'holds' in task_pred:
				print("GRABED", grabbed)

				current_object = task_pred.split('_')[1]
				if len(grabbed) > 0 and grabbed[0] in current_object:
					last_completed[task_pred] = 1
					total_completed[task_pred] = 1
				else:
					total_completed[task_pred] = 0
					last_completed[task_pred] = 0

			else:
				current_object = task_pred.split('_')[1]
				if obj_id_pred in visible_ids:
					last_completed[task_pred] = len([obj_name for obj_name in ids_curr if obj_name in current_object])
				total_completed[task_pred] = len([obj_name for obj_name in ids_curr if obj_name in current_object])


	# print(curr_task['task_goal'])
	other_info['task_preds'] = []

	# Generate string
	for task_pred, count in curr_task['task_goal'][0].items():
		if count > 0 and 'sit' not in task_pred:
			completed = last_completed[task_pred]
			completed_all = total_completed[task_pred]
			task_pred_name = task_pred
			# if task_pred == 'holds_book_character':
			# task_pred_name = 'hold_book'
			task_pred_name = task_pred.split('_')
			last_id = task_pred_name[-1]
			last_class = id2node[int(last_id)]['class_name']
			task_pred_name = f'put {task_pred_name[1]} {task_pred_name[0]}to the {last_class}.{task_pred_name[2]}'
			num_preds_needed += count
			num_preds_done += min(completed, count)
			num_preds_done_all += min(completed_all, count)

			other_info['task_preds'].append('{}: {}/{}'.format(task_pred_name, completed, count))

	if num_preds_done_all == num_preds_needed or record_step_counter >= 250:
		other_info['task_finished'] = '1'
	else:
		other_info['task_finished'] = '0'

	other_info['step_str'] = '{}/250'.format(record_step_counter)

	other_info['total_completed_str'] = '{}/{}'.format(num_preds_done, num_preds_needed)

	other_info['task_id_str'] = '{}/{}'.format(task_index+1, len(args.task_group))

	other_info['goal_id'] = obj_id_pred

	other_info['dialogue_history'] = '\n-------------------------------\n'.join(dialogue_history[::-1])

	visible_ids = [v[0] for v in other_info['visible_objects']]

	# other_info['not_visible_objects'] = [[216, 'cabinet', 216, 'container', '0', False, -1, 210, 'bedroom']]

	# other_info['not_visible_objects'] = [i for i in previous_belief if i[0] not in visible_ids]
	def append_list_0(a,b):
		l = copy.deepcopy(a)
		for i in b:
			if i[0] not in [j[0] for j in l]:
				l.append(i)
		return l
	def remove_list_0(a,roomid):
		return [i for i in a if i[7] != roomid]
	def append_list_00(a,b):
		l = copy.deepcopy(a)
		for i in b:
			if i[0][0] not in [j[0][0] for j in l]:
				l.append(i)
		return l
	def remove_list_00(a,roomid):
		return [i for i in a if i[0][7] != roomid]


	previous_belief = append_list_0(other_info['visible_objects'], remove_list_0(previous_belief, current_room))

	previous_belief_action = append_list_00(object_action, remove_list_00(previous_belief_action, current_room))

	# print(f"{previous_belief=}")

	def remove_notgrab_0(l, fg):
		l = [i for i in l if i[3] == "grab" and not any([i[0] == e['from_id'] for e in fg["edges"] if e['to_id'] not in rooms_id])]
		return l

	other_info['not_visible_objects'] = remove_list_0(previous_belief, current_room)

	info.update({'object_action':object_action, 'other_info': other_info})

	def remove_notgrab_00(l, fg):
		l = [i for i in l if i[0][3] == "grab" and not any([i[0][0] == e['from_id'] for e in fg["edges"] if e['to_id'] not in rooms_id])]
		return l


	info['not_visible_object_action'] = remove_list_00(previous_belief_action, current_room)


	images = refresh_image(g, [obj_id_pred])#sjm not needed, delete it.


	if record_graph_flag:
		if not os.path.exists(graph_save_dir):
			os.makedirs(graph_save_dir)
		with open(os.path.join(graph_save_dir, 'file_{}.json'.format(record_step_counter)), 'w') as f:
			# g_clean = {
			#	 'nodes': [vh_tools.reduced_node_info(node) for node in graph['nodes'] if node['id'] not in bad_class_name_ids],
			#	 'edges': [edge for edge in graph['edges'] if edge['from_id'] not in bad_class_name_ids and edge['from_id'] not in bad_class_name_ids]
			# }
			file_info = {
				'instruction': script,
				'time': time.time() - time_start,
				'predicates': total_completed,
				'agent_info': agent_info,
				'graph': savegraph,
			}
			json.dump(file_info, f)
		if command['instruction'] != 'refresh':
			record_step_counter += 1

	info['all_done'] = False
	return images, info

'''
def reset(scene, init_graph=None, init_room=[]):
	global lc
	global instance_colors
	global image_top
	global next_helper_action
	global instance_colors_reverse
	global bad_class_name_ids, record_step_counter, task_index, graph_save_dir, curr_task, task_index_shuffle
	global last_completed, current_goal
	global goal_spec
	global extra_agent
	global graph
	global extra_agent_list
	global id2classid, class2numobj, id2node
	global last_instr_main

	class2numobj = {}
	id2classid = {}
	id2node = {}

	record_step_counter = 1
	task_index = task_index + 1
	all_done = False
	if task_index >= len(args.task_group):
		return None, {'all_done': True}
		print('All tasks in task_group {} are finished.'.format(args.task_group))
		task_index = task_index - 1
	#print(task_index)
	temp_task_id = int(args.task_group[task_index_shuffle[task_index]])
	graph_save_dir = 'record_graph/{}/task_{}/time.{}'.format(args.exp_name, temp_task_id, time_str)

	#### For debug  ####
	pkl_file = save_pkl_file
	with open(pkl_file, 'rb') as f:
		env_task_set = pkl.load(f)
	print(len(env_task_set), temp_task_id)
	curr_task = env_task_set[temp_task_id]
	add_goal_class(curr_task)
	# print(temp_task_id)
	scene = curr_task['env_id']
	init_graph = curr_task['init_graph']
	init_graph = clean_graph(init_graph)
	init_room = curr_task['init_rooms']
	#### Debug code finished.

	last_completed = {}
	last_instr_main = None
	env.reset(scene)
	if init_graph is not None:

		s, m = comm.expand_scene(init_graph)
		print("EXPNAD", m)
	
	comm.add_character('Chars/Female1', initial_room=init_room[0])
	extra_agent_name = extra_agent_list[task_index_shuffle[task_index]]

	print("TASK", extra_agent_name, temp_task_id)
	if extra_agent_name != "none":
		
		sys.path.append('../../')
		import agents
		print(agents)
		print(sys.path)
		comm.add_character('Chars/Male1', initial_room=init_room[1])

	s, g = comm.environment_graph()
	# ipdb.set_trace()
	graph = g
	images = refresh_image(current_graph=g)
	# images = [image]
	#image_top = images[0]
	

	if extra_agent_name == "nopa":
		extra_agent = agents.NOPA_agent(agent_id=2,
									   char_index=1,
									   max_episode_length=5,
									   num_simulation=100,
									   max_rollout_steps=5,
									   c_init=0.1,
									   c_base=1000000,
									   num_samples=20,
									   num_processes=1,
									   seed=temp_task_id)
		gt_graph = g
		#print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
		task_goal = None
		container_id = int(list(curr_task['task_goal'][0].keys())[0].split('_')[-1])

		observed_graph = vh_tools.get_visible_graph(g, agent_id=2, full_obs=True)
		extra_agent.reset(observed_graph, gt_graph, container_id, task_goal)
	
	if extra_agent_name == "hp_gp":
		extra_agent = agents.MCTS_agent(agent_id=2,
									   char_index=1,
									   max_episode_length=5,
									   num_simulation=100,
									   max_rollout_steps=5,
									   c_init=0.1,
									   c_base=1000000,
									   num_samples=20,
									   num_processes=1,
									   seed=temp_task_id)
		gt_graph = g
		#print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
		task_goal = None
		container_id = int(list(curr_task['task_goal'][0].keys())[0].split('_')[-1])

		# observed_graph = vh_tools.get_visible_graph(g, agent_id=2, full_obs=True)
		obs = env.get_observation(agent_id=1, obs_type='partial')
		extra_agent.reset(obs, gt_graph, container_id, task_goal)

	if extra_agent_name == "hp_random":
		extra_agent = agents.HP_random_agent(agent_id=2,
											   char_index=1,
											   max_episode_length=5,
											   num_simulation=100,
											   max_rollout_steps=5,
											   c_init=0.1,
											   c_base=1000000,
											   num_samples=20,
											   num_processes=1,
											   seed=temp_task_id)
		gt_graph = g
		#print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
		task_goal = None
		container_id = int(list(curr_task['task_goal'][0].keys())[0].split('_')[-1])

		observed_graph = vh_tools.get_visible_graph(g, agent_id=2, full_obs=True)
		extra_agent.reset(observed_graph, gt_graph, container_id, task_goal)

	if extra_agent_name  == "random_goal":
		print("RANDOM")
		extra_agent = agents.MCTS_agent(agent_id=2,
									   char_index=1,
									   max_episode_length=5,
									   num_simulation=100,
									   max_rollout_steps=5,
									   c_init=0.1,
									   c_base=1000000,
									   num_samples=1,
									   num_processes=1,
									   seed=temp_task_id)
		gt_graph = g
		#print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
		task_goal = None
		observed_graph = vh_tools.get_visible_graph(g, agent_id=2)
		extra_agent.reset(observed_graph, gt_graph, task_goal)
		goal_spec = vh_tools.get_random_goal(g, temp_task_id)
		print("GOAL", goal_spec, temp_task_id)

	elif extra_agent_name  == 'rl_mcts':
		from util import utils_rl_agent
		trained_model_path = 'ADD HERE the training path'
		model_path = (f'{trained_model_path}/trained_models/env.virtualhome/'
					  'task.full-numproc.5-obstype.mcts-sim.unity/taskset.full/agent.hrl_mcts_alice.False/'
					  'mode.RL-algo.a2c-base.TF-gamma.0.95-cclose.0.0-cgoal.0.0-lr0.0001-bs.32_finetuned/'
					  'stepmcts.50-lep.250-teleport.False-gtgraph-forcepred/2000.pt')
		arg_dict = {
				'name': 'model_args',
				'evaluation': True,
				'max_num_objects': 150,
				'hidden_size': 128,
				'init_epsilon': 0,
				'base_net': 'TF',
				'teleport': False,
				'model_path': model_path,
				'num_steps_mcts': 40
		}
		Args = namedtuple('agent_args', sorted(arg_dict))
		model_args = Args(**arg_dict)
		graph_helper = utils_rl_agent.GraphHelper(max_num_objects=150,
												  max_num_edges=10, current_task=None,
												  simulator_type='unity')
		extra_agent = agents.HRL_agent(agent_id=2,
									   char_index=1,
									   args=model_args,
									   graph_helper=graph_helper)
		curr_model = torch.load(model_args.model_path)[0]
		extra_agent.actor_critic.load_state_dict(curr_model.state_dict())
		gt_graph = g
		#print([node for node in gt_graph['nodes'] if node['id']  in [1,2]])
		task_goal = None
		observed_graph = vh_tools.get_visible_graph(g, agent_id=2)
		extra_agent.reset(observed_graph, gt_graph, task_goal)

	
	elif extra_agent_name  == 'none':
		extra_agent = None
		goal_spec = None


	if not os.path.exists(graph_save_dir):
		os.makedirs(graph_save_dir)
	
	with open(os.path.join(graph_save_dir, 'init_graph.json'), 'w') as f:
		file_info = {
			'graph': graph,
			'instruction': "START",
			'extra_agent': str(extra_agent_name),
		}
		json.dump(file_info, f)

	# bad_class_name_ids = [node['id'] for node in graph['nodes'] if node['class_name'] in vh_tools.get_classes_ignore()]
	# for node in graph['nodes']:
	#	 if node['class_name'] not in class2numobj:
	#		 class2numobj[node['class_name']] = 0
	#	 class2numobj[node['class_name']] += 1
	#	 id2classid[node['id']] = class2numobj[node['class_name']]

	return images, {'image_top': get_top_image(), 'all_done': all_done}
'''

def reset(scene):
	global image_top
	global env
	global lc
	global instance_colors
	global current_image
	global images
	global prev_images
	global graph
	global id2node
	global aspect_ratio
	global bad_class_name_ids
	global curr_task
	global last_completed
	global previous_belief
	global previous_belief_action
	global extra_agent
	global next_helper_action
	global time_start
	global record_graph_flag # False
	global graph_save_dir
	global record_step_counter
	global dialogue_history
	global task_index # for indexing task_id in task_group
	global task_index_shuffle
	global last_instr_main
	global current_goal
	os.system(f'kill -9 $(lsof -t -i:{args.portvh})')

	image_top = None
	# comm = None
	env = None
	lc = None
	instance_colors = None
	current_image = None
	images = None
	prev_images = None
	graph = None
	id2node = None
	aspect_ratio = 9./16.
	bad_class_name_ids = []
	curr_task = None
	last_completed = {}
	previous_belief = []
	previous_belief_action = []
	extra_agent = None
	next_helper_action = None

	# parameters for record graph
	time_start = None
	record_graph_flag = True # False
	#vis_graph_flag = True
	graph_save_dir = None

	record_step_counter = 1
	dialogue_history = []

	# Contains a mapping so that objects have a smaller id. One unique pr object+instance
	# instead of instance. glass.234, glass.267 bcecome glass.1 glass.2
	# class2numobj = {}
	last_instr_main = None

	current_goal = {}


	task_index = task_index + 1
	all_done = False
	print('task_index', task_index)
	print('args.task_group', args.task_group)
	if task_index >= len(args.task_group):
		return None, {'all_done': True}
		print('All tasks in task_group {} are finished.'.format(args.task_group))
		task_index = task_index - 1
	#print(task_index)
	temp_task_id = int(args.task_group[task_index_shuffle[task_index]])
	graph_save_dir = 'record_graph/{}/task_{}/trial_{}/time.{}'.format(args.extra_agent, temp_task_id, args.trial, time_str)
	# comm = comm_unity.UnityCommunication(file_name=args.executable_file, port=str(args.portvh), no_graphics=True)
	pkl_file = save_pkl_file
	with open(pkl_file, 'rb') as f:
		env_task_set = pkl.load(f)

	two_agent = args.extra_agent != 'none'

	if not two_agent:
		envarg = {'num_agents': 1,
				  'max_episode_length': 250,
				  'env_task_set':env_task_set,
				  'observation_types': ['partial'],
				  'agent_goals': ['full'],
				  'use_editor': False,
				  'base_port': int(args.portvh),
				  'port_id': 0,
				  'executable_args': {
						'file_name': args.executable_file,
						'no_graphics': True,
				  		},
				  }
	else:
		envarg = {'num_agents': 2,
				  'max_episode_length': 250,
				  'env_task_set':env_task_set,
				  'observation_types': ['partial', 'partial'],
				  'agent_goals': ['full', 'LLM' if 'LLM' in args.extra_agent else 'full'],
				  'use_editor': False,
				  'base_port': int(args.portvh),
				  'port_id': 0,
				  'executable_args': {
						'file_name': args.executable_file,
						'no_graphics': True,
				  		},
				  }
	env = UnityEnvironment(**envarg)
	# env = UnityEnvironment(num_agents=2,max_episode_length=250,env_task_set=env_task_set,observation_types=['partial', 'partial'],agent_goals=None,use_editor=False,base_port=int(args.portvh))
	# env.comm = comm

	curr_task = env_task_set[temp_task_id]
	# ipdb.set_trace()
	ob = None
	while ob is None:
		ob = env.reset(curr_task['init_graph'], temp_task_id)
		#----------------------reset extra agent----------------------
		# '''

		if args.extra_agent == 'MCTS_comm':
			extra_agent = agents.MCTS_agent(agent_id=2,
											char_index=1,
											max_episode_length=5,
											num_simulation=100,
											max_rollout_steps=5,
											c_init=0.1,
											c_base=1000000,
											num_samples=20,
											num_processes=1,
                                            logging=True,
											seed=temp_task_id,
											belief_comm=False,
											opponent_subgoal='comm',
											satisfied_comm=True)
		elif args.extra_agent == 'LLM_comm':
			args.communication = True
			args.prompt_template_path = "../LLM/prompt_com.csv"
			args.cot= True
			extra_agent = agents.LLM_agent(agent_id=2,
										   char_index=1,
										   args=args)#sjm args to be completed
		elif args.extra_agent == 'LLM':
			args.communication = False
			args.prompt_template_path = "../LLM/prompt_nocom.csv"
			args.cot= True
			extra_agent = agents.LLM_agent(agent_id=2,
										   char_index=1,
										   args=args)#sjm args to be completed

	if args.extra_agent != 'none':
		# if 'LLM_vision' in extra_agent.agent_type:
		# 	extra_agent.reset(ob[1], env.all_containers_name, env.all_goal_objects_name, env.all_room_name, env.goal_spec[1])
		# elif 'vision' in extra_agent.agent_type:
		# 	extra_agent.reset(ob[1], env.full_graph, env.task_goal, env.all_room_name, env.all_containers_name, env.all_goal_objects_name, seed=extra_agent.seed)
		if 'MCTS' in extra_agent.agent_type:
			extra_agent.reset(ob[1], env.full_graph, env.task_goal, seed=extra_agent.seed)
		elif 'LLM' in extra_agent.agent_type:
			extra_agent.reset(ob[1], env.all_containers_name, env.all_goal_objects_name, env.all_room_name, env.room_info, env.goal_spec[1])
		else:
			raise ValueError(f"Not available agent type {extra_agent.agent_type}")


	time_start = time.time()
	return {'image_top': get_top_image(), 'all_done': all_done}


def get_helper_action(goal_spec, previous_main_action, num_steps):
	curr_obs = env.get_observation(agent_id=1, obs_type='partial')
	# curr_obs = vh_tools.get_visible_graph(gt_graph, agent_id=2, full_obs=False)
	# curr_obs = vh_tools.get_obs(gt_graph, agent_id=2, obs_type='partial')
	print('({})'.format(extra_agent.agent_type), '--')

	if 'MCTS' in extra_agent.agent_type:
		#import cProfile, pstats
		#from pstats import SortKey
		#with cProfile.Profile() as pr:
		# command_other = extra_agent.get_action(curr_obs, goal_spec, previous_main_action, num_steps)[0] #old ver, sjm change to merge interface
		command_other, info = extra_agent.get_action(curr_obs, goal_spec)
		info.pop('belief')
		info.pop('belief_graph')
		if command_other is not None:
			command_other = command_other.replace('[walk]', '[walktowards]')
        
		#pstats.Stats(pr).sort_stats(SortKey.CUMULATIVE).print_stats()
		#import ipdb
		#ipdb.set_trace()
	elif 'LLM' in extra_agent.agent_type:
		command_other, info = extra_agent.get_action(curr_obs, goal_spec)
		if 'LLM' in info:
			print(info['LLM'])
	else:
		raise Exception
		# The ids iwth which we can do actions
		new_goal_spec = {}
		for pred, ct in goal_spec.items():
			if 'holds' in pred or ct == 0:
				continue
			required = True
			reward = 0
			count = ct
			new_goal_spec[pred] = [count, required, reward]
		action_space = [node['id'] for node in curr_obs['nodes']]
		action, info = extra_agent.get_action(curr_obs, new_goal_spec, action_space_ids=action_space, full_graph=None)
		command_other = action
		print('------------')
		print("EXTRA AGENT")
		print(action, info['predicate'], goal_spec)
		print('------------')

	# command_other = '[send_message] <this is just used to test send message correctness>'
	# command_other = '[walktowards] <kitchen> (167)'
	return command_other, info


def get_top_image():
	global image_top
	return image_top

def refresh_image(current_graph, curr_goal_id=[]):
	# global prev_images
	# print("GOAL", curr_goal_id)
	# ipdb.set_trace()
	visible_ids = vh_tools.get_objects_visible(None, current_graph, ignore_bad_class=True, full_obs=False)
	char_ids = [node['id'] for node in current_graph['nodes'] if node['id'] in visible_ids and node['class_name'] == 'character']
	curr_goal_id = [cgid for cgid in curr_goal_id if cgid in visible_ids]
	fig = plot_graph_2d(current_graph, visible_ids=visible_ids, action_ids=[], char_id=char_ids, goal_ids=curr_goal_id, display_furniture=False)
	plt.axis('off')

	fig.tight_layout(pad=0)
	fig.canvas.draw()

	image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

	image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	if curr_task['env_id'] == 6:
		h = int(0.2 * image_from_plot.shape[0])
		image_from_plot = image_from_plot[h:-h, :]
	# image_top = np.transpose(image_top, (1,0,2))
	# else:
	# h = int(0. * image_from_plot.shape[0])
	# image_from_plot = image_from_plot[h:-h, :]
	plt.close()
	return [np.transpose(image_from_plot, (1,0,2))[:, ::-1, :]]


@app.route('/')
def output():
	# serve index template

	return render_template('index_headless.html', name='Joe', show_modal=args.showmodal)


@app.route('/querymaskid', methods = ['POST'])
def get_mask_id():
	global instance_colors_reverse
	data = request.get_json(silent=False)
	obj_id = data['obj_id']

	visible_ids = vh_tools.get_objects_visible(None, graph, ignore_bad_class=True, full_obs=False)
	char_ids = [node['id'] for node in graph['nodes'] if node['id'] in visible_ids and node['class_name'] == 'character']

	fig = plot_graph_2d(graph, visible_ids=visible_ids, action_ids=[obj_id], char_id=char_ids, goal_ids=[], display_furniture=False)
	plt.axis('off')

	fig.tight_layout(pad=0)
	fig.canvas.draw()

	image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	if curr_task['env_id'] == 6:
		h = int(0.2 * image_from_plot.shape[0])
		image_from_plot = image_from_plot[h:-h, :]

	current_images = [np.transpose(image_from_plot, (1,0,2))[:, ::-1, :]]
	# current_images = [image_from_plot]
	img_64s = [convert_image(current_image) for current_image in current_images]
	result = {}
	result.update({'img': str(img_64s[0].decode("utf-8"))})
	result = json.dumps(result)


	return result


@app.route('/record_graph_flag', methods = ['POST'])
def set_record_graph_flag():
	global record_graph_flag
	record_graph_flag = not record_graph_flag
	return {'record_graph_flag': record_graph_flag}

@app.route('/receiver', methods = ['POST'])
def worker():
	start_t = time.time()
	# read json + reply
	data = request.get_json(silent=False)
	t1 = time.time()
	# print('t1 - start_t: ', t1 - start_t)
	current_images, out = send_command(data)
	t2 = time.time()
	# print('t2 - t1: ', t2 - t1)
	#current_images  = [current_images[0]]
	# print(current_images)
	# print("HERE")

	result = {'resp': out}
	if out['all_done']:
		result = json.dumps(result)
		return result
	# print(current_images)
	# result.update({'plot_top': str(current_images)})
	img_64s = [convert_image(current_image[:,:,:]) for current_image in current_images]
	result.update({'img': str(img_64s[0].decode("utf-8"))})


	#print(result['img'][:5])
	#print(result['resp']['image_top'][:5])

	result = json.dumps(result)
	#cv2.imwrite('static/curr_im.png', current_images[0])
	#result = json.dumps({'resp': 'current.png'})
	end_t2 = time.time()
	return result

if __name__ == '__main__':
	# run!
	os.system(f'kill -9 $(lsof -t -i:{args.portvh})')

	# global graph_save_dir
	task_index_shuffle = list(range(len(args.task_group)))
	print(task_index_shuffle)
	# code2agent = {
	#		 'none': 'none',
	#		 'B1': 'nopa',
	#		 'B2': 'hp_gp',
	#		 'B3': 'hp_random',
	# }
	# extra_agent_list = [code2agent[name] for name in args.extra_agent]
	# print(extra_agent_list)
	random.shuffle(task_index_shuffle)

	#random.Random(args.exp_name).shuffle(task_index_shuffle)
	print('args.task_group',args.task_group)
	print('task_index_shuffle',task_index_shuffle)
	now = datetime.now()
	date_time = now.strftime("%m.%d.%Y-%H.%M.%S")
	time_str = str(date_time)
	reset(None)


	time_start = time.time()
	if args.deployment == 'local':
		app.run(host='localhost', port=str(args.portflask))
	else:
		app.run(host='0.0.0.0', port=str(args.portflask))

