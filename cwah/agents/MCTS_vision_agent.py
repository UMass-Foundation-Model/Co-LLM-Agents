import cv2
import re
import numpy as np
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import pickle
import numpy as np
import open3d as o3d


from . import belief
from . import utils
from . import vision_pipeline
from envs.graph_env import VhGraphEnv
#
from MCTS import *
from agents.MCTS_agent import *

import sys
sys.path.append('..')
from utils import utils_environment as utils_env

class MCTS_vision_agent():
    """
    MCTS for vision agent(s).
    """
    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, recursive=False,
                 num_samples=1, num_processes=1, logging=False, logging_graphs=False, seed=None, belief_comm=False, opponent_subgoal='None', gt_seg=True, satisfied_comm = False, verbose = False):
        
        self.agent_type = 'MCTS_vision'
        self.recursive = recursive
        self.config = vision_pipeline.agent_vision_config(
                agent_type = 'MCTS_vision',
                char_index = char_index,
                agent_id = agent_id,
                gt_seg = gt_seg,
                )
        self.agent_id = agent_id
        self.char_index = char_index
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.opponent_subgoal = opponent_subgoal
        self.belief_comm = belief_comm
        self.satisfied_comm = satisfied_comm
        self.seed = random.randint(0,100),
        self.logging = logging
        self.logging_graphs = logging_graphs
        self.gt_seg = gt_seg
        self.verbose = verbose
        self.c_init = c_init
        self.c_base = c_base
        self.relative_container = []
        self.relative_goal = []
        self.relative_room = []
        self.plan = None

    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph['edges']:
            key = (edge['from_id'], edge['to_id'])
            if key not in edge_dict:
                edge_dict[key] = [edge['relation_type']]
                new_edges.append(edge)
            else:
                if edge['relation_type'] not in edge_dict[key]:
                    edge_dict[key] += [edge['relation_type']]
                    new_edges.append(edge)
        graph['edges'] = new_edges
        return graph

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_action(self, obs, goal_spec, opponent_subgoal=None):
        self.vision_pipeline.deal_with_obs(obs, self.last_action)
        graph_obs = self.vision_pipeline.get_graph()

        '''dividing line, below is the code for decision making, and above is the code for observation dealing'''
        # Update belief msg, subgoal msg, and navigation module.

        if obs['messages'] != None and len(obs['messages']) > 1 and obs['messages'][1 - self.char_index]:
            msg = eval(utils.language_to_MCTS_convert(obs['messages'][1 - self.char_index]))
            if self.char_index == 0 and obs['messages'][0] != None and 'S' in msg.keys(): 
                msg.pop('S') # do not recurrent deliever subgoal
            # check message state
            if self.opponent_subgoal == 'comm':
                if 'S' in msg.keys(): self.received_opponent_subgoal = msg['S']
            if self.belief_comm:
                if 'B' in msg.keys(): self.belief.receive_belief_new(msg['B'])

        if self.last_room != obs['current_room']:
            assert 'walktowards' in self.last_action
            self.navigation_goal = int(self.plan[1].split(' ')[-1][1: -1])
            to_room_id = int(self.plan[0].split(' ')[-1][1: -1])
            if self.navigation_goal in self.vision_pipeline.object_info.keys() or to_room_id != obs['current_room']:
                self.need_navigation = False
            else: 
                self.need_navigation = True
        # Judge when entering a room, if the agent cannot see it. He needs navigation.

        send_belief_flag = False # Whether to send belief update
        mcts_flag = True # Whether to use MCTS
        msg_to_send = "" # Message to send
        subgoal = self.last_subgoal  
        info = {}
        if self.opponent_subgoal in ['None', 'comm']: opponent_subgoal = None

        # Since we go out of the room, we need to send the belief update before we update the map
        if self.belief_comm and obs['current_room'] != self.last_room:
            upd_belief_info = self.belief.delta_record_belief_new()
            total_value = 0
            filter_upd_info = []
            for one_info in upd_belief_info:
                filter_one_info = []
                for obj in one_info[1]:
                    if obj not in self.vision_pipeline.with_character_id:
                        filter_one_info.append(obj)
                        total_value += 4
                filter_upd_info.append((one_info[0], filter_one_info))
                total_value += 1
            if total_value >= 5: # Balance between communication cost and information gain
                msg_to_send += f"'B':{filter_upd_info},"
                send_belief_flag = True
                self.belief.update_record_belief()

        self.sample_belief(graph_obs)
        self.sim_env.reset(self.previous_belief_graph, None)
        
        if self.need_navigation and send_belief_flag == False:
            # Explore the room
            # if self.navigation_goal in self.vision_pipeline.visiable_ids:
            #    self.need_navigation = False
            #  keep navigation module same. LLM do not know the navigation goal (what to explore)
            if (self.last_location != obs['location'] or 'message' in self.last_action) and self.remain_rotation_cnt in [0, 11] and self.keep_move < 10:
                # Turn left may lead to position change
                action = f'[walktowards] <{self.vision_pipeline.object_info[obs["current_room"]]["class_name"]}> ({obs["current_room"]})' # keep going 
                self.remain_rotation_cnt = 11 # 11 * 30 = 330 degrees
                self.keep_move += 1
                mcts_flag = False
            else:
                action = '[TurnLeft]'
                self.remain_rotation_cnt -= 1
                self.keep_move = 0
                if self.remain_rotation_cnt <= 0: self.need_navigation = False
                mcts_flag = False
          
        if mcts_flag and send_belief_flag == False: # Use oringinal MCTs, rather than navigation
            if self.opponent_subgoal == 'comm': 
                if self.received_opponent_subgoal is not None:
                    opponent_subgoal = self.received_opponent_subgoal[:-1]
                else:
                    opponent_subgoal = None
            self.keep_move = 0
            self.remain_rotation_cnt = 0
            # Clear all the parameter in the last navigation module.
            self.plan, root_node, subgoals = get_plan(None, None, None, self.sim_env, self.mcts, 0, goal_spec, None, self.last_subgoal, self.last_action, opponent_subgoal, verbose=self.verbose)
            if self.opponent_subgoal == 'comm' and self.received_opponent_subgoal is not None and self.received_opponent_subgoal[-1] == '0': 
                self.received_opponent_subgoal = None
                # 0 means plan to do, 1 means successfully find it
                # If finding it, repeatly send the same subgoal until success.
            if len(self.plan) > 0:
                action = self.plan[0]
                action = action.replace('[walk]', '[walktowards]')
                if '[open]' in action:
                    goal_obj = int(re.findall(r'\d+', action)[0])
                    if goal_obj not in self.vision_pipeline.see_this_step or 'OPEN' in self.vision_pipeline.object_info[goal_obj]['states']:
                        action = action.replace('[open]', '[walktowards]')
                # If the action is open, then check whether the object is in the vision. If not, then change to walktowards
            else:
                action = None
            subgoal = subgoals[0] if len(subgoals) > 0 else None
        
        if self.satisfied_comm: # Put the part here just to log some info
            # If previous action is putback or putin, then send this info to other. 
            last_action_name = utils_env.get_action_name(self.last_action)
            if last_action_name == "putback" or last_action_name == "putin":
                roomname = [edge['to_id'] for edge in self.sim_env.state['edges'] if edge['from_id'] == self.agent_id and edge['relation_type'] == 'INSIDE'][0]
                objectname = utils_env.get_id(self.last_action)
                containername = utils_env.get_last_id(self.last_action)
                if last_action_name == "putback":
                    msg_to_send_edges = f"{{'from_id': {objectname}, 'to_id': {containername}, 'relation_type': 'ON', 'room_id': {roomname}}}"
                if last_action_name == "putin":
                    msg_to_send_edges = f"{{'from_id': {objectname}, 'to_id': {containername}, 'relation_type': 'INSIDE', 'room_id': {roomname}}}"
                msg_to_send += f"'E':{msg_to_send_edges},"

        if self.opponent_subgoal == 'comm':
            if self.previous_sent_subgoal != subgoal and subgoal is not None:
                decide_send_this_turn = (self.step >= self.char_index) # Avoid stuck
                if decide_send_this_turn:
                    obj = int(subgoal.split('_')[1])
                    if obj in self.vision_pipeline.object_info.keys(): msg_to_send += f"'S':'{str(subgoal) + '_1'}',"
                    else: msg_to_send += f"'S':'{str(subgoal) + '_0'}',"
                    self.previous_sent_subgoal = subgoal

        if msg_to_send != "":
            msg_to_send = '{' + msg_to_send + '}'
            text_message = utils.MCTS_to_language_convert(msg_to_send, id_to_name_api = obs['get_class_name'])
            action = f'[send_message] <{text_message}>'

        self.step += 1
        self.last_action = action
        self.last_subgoal = subgoal
        self.last_location = obs['location']
        self.last_room = obs['current_room']

        info['subgoals'] = subgoal
        return action, info

    def reset(self, obs, gt_graph, task_goal, room_name = [], relative_container_name = [], relative_goal_name = [], seed=0, simulator_type='python', is_alice=False):
        self.vision_pipeline = vision_pipeline.Vision_Pipeline(self.config, obs)
        self.step = 0
        self.keep_move = 0
        self.task_goal = task_goal

        self.relative_room = room_name
        self.relative_container = relative_container_name 
        self.relative_goal = relative_goal_name
        self.received_opponent_subgoal = None

        self.last_location = np.asarray([0, 0, 0])
        self.remain_rotation_cnt = 0

        self.need_navigation = False
        self.previous_sent_subgoal = None
        self.previous_belief_graph = None
        self.last_action = None
        self.last_subgoal = None
        self.last_room = obs['current_room']
        self.plan = None

        relative_nodes_id = [node['id'] for node in gt_graph['nodes'] if node['class_name'] in self.relative_container + self.relative_goal + self.relative_room + ['character']]
        gt_graph = {
            'nodes': [node for node in gt_graph['nodes'] if node['id'] in relative_nodes_id],
            'edges': [edge for edge in gt_graph['edges'] if edge['from_id'] in relative_nodes_id and edge['to_id'] in relative_nodes_id]
        }

        self.belief = belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed, task_goal = task_goal)
        self.belief.sample_from_belief()
        self.mcts = MCTS(self.sim_env, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base, seed=seed, verbose=self.verbose)