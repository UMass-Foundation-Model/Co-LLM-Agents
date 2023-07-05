from .base_environment import BaseEnvironment
from utils import utils_environment as utils

import sys
import os



curr_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f'{curr_dir}/../../virtualhome/')
sys.path.append(f'{curr_dir}/../../vh_mdp/')

from vh_graph.envs import belief, vh_env
import pdb
import random
import numpy as np
import copy

class PythonEnvironment(BaseEnvironment):


    def __init__(self,
                 num_agents=2,
                 max_episode_length=200,
                 env_task_set=None,
                 observation_types=None,
                 agent_goals=None,
                 output_folder=None,
                 seed=123):

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.steps = 0
        self.env_id = None
        self.max_ids = {}

        self.pythnon_graph = None
        self.env_task_set = env_task_set

        self.num_agents = num_agents
        self.max_episode_length = max_episode_length

        self.output_folder = output_folder


        if observation_types is not None:
            self.observation_types = observation_types
        else:
            self.observation_types = ['mcts' for _ in range(num_agents)]

        if agent_goals is not None:
            self.agent_goals = agent_goals
        else:
            self.agent_goals = ['full' for _ in range(num_agents)]



        self.task_goal, self.goal_spec = {0: {}, 1: {}}, {0: {}, 1: {}}

        self.changed_graph = False
        self.rooms = None
        self.id2node = None
        self.offset_cameras = None


        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)

    def reward(self):
        reward = 0.
        done = True
        # TODO: we should specify goal per agent maybe
        satisfied, unsatisfied = utils.check_progress(self.get_graph(), self.goal_spec[0])
        for key, value in satisfied.items():
            preds_needed, mandatory, reward_per_pred = self.goal_spec[0][key]
            # How many predicates achieved
            value_pred = min(len(value), preds_needed)
            reward += value_pred * reward_per_pred

            if mandatory and unsatisfied[key] > 0:
                done = False

        # print(satisfied)
        return reward, done, {}

    def step(self, action_dict):
        new_action_dict = {char_id: action for char_id, action in action_dict.items() if action is not None}
        self.env.step(new_action_dict)
        self.changed_graph = True


        reward, done, info = self.reward()
        obs = self.get_observations()
        self.steps += 1
        info['finished'] = done
        info['graph'] = self.get_graph()
        if self.steps == self.max_episode_length:
            done = True
        # if done:
        #     pdb.set_trace()
        return obs, reward, done, info

    def python_graph_reset(self, graph):
        new_graph = utils.inside_not_trans(graph)
        self.python_graph = new_graph
        self.env.reset(new_graph, self.task_goal)
        self.env.to_pomdp()

    def get_goal(self, task_spec, agent_goal):

        if agent_goal == 'full':
            task_spec_new = {goal_name: [cnt_val, True, 0] for goal_name, cnt_val in task_spec.items()}
            return task_spec_new
        elif agent_goal == 'grab':
            candidates = [x.split('_')[1] for x,y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']]
            object_grab = random.choice(candidates)
            # print('GOAL', candidates, object_grab)
            return {'holds_'+object_grab+'_'+'1': [1, True, 10], 'close_'+object_grab+'_'+'1': [1, False, 0.1]}
        elif agent_goal == 'put':
            pred = random.choice([x for x, y in task_spec.items() if y > 0 and x.split('_')[0] in ['on', 'inside']])
            object_grab = pred.split('_')[1]
            return {
                pred: [1, True, 60],
                'holds_' + object_grab + '_' + '1': [1, False, 2],
                'close_' + object_grab + '_' + '1': [1, False, 0.05]

            }
        else:
            raise NotImplementedError

    def reset(self, environment_graph=None, task_id=None):

        # Make sure that characters are out of graph, and ids are ok
        if task_id is None:
            env_task = random.choice(self.env_task_set)
        else:
            env_task = self.env_task_set[task_id]

        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']

        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']



        # TODO: in the future we may want different goals
        self.goal_spec = {agent_id: self.get_goal(self.task_goal[agent_id], self.agent_goals[agent_id])
                          for agent_id in range(self.num_agents)}
        print("Goal: ", self.goal_spec)

        if environment_graph is None:
            environment_graph = env_task['init_graph']


        if self.init_rooms[0] not in ['kitchen', 'bedroom', 'livingroom', 'bathroom']:
            rooms = random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        else:
            rooms = list(self.init_rooms)

        environment_graph = copy.deepcopy(environment_graph)
        for i in range(self.num_agents):
            new_char_node = {
                'id': i+1,
                'class_name': 'character',
                'states': [],
                'category': 'Characters',
                'properties': []
            }
            room_name = rooms[i]
            #try:
            room_id = [node['id'] for node in environment_graph['nodes'] if node['class_name'] == room_name][0]
            #except:
            #    pdb.set_trace()
            environment_graph['nodes'].append(new_char_node)
            environment_graph['edges'].append({'from_id': i+1, 'relation_type': 'INSIDE', 'to_id': room_id})


        self.python_graph_reset(environment_graph)
        self.rooms = [(node['class_name'], node['id']) for node in environment_graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in environment_graph['nodes']}

        obs = self.get_observations()
        self.steps = 0
        return obs

    def get_graph(self):
        out_graph = self.env.state
        return out_graph

    def get_observations(self):
        dict_observations = {}
        for agent_id in range(self.num_agents):
            obs_type = self.observation_types[agent_id]
            dict_observations[agent_id] = self.get_observation(agent_id, obs_type)
        return dict_observations

    def get_action_space(self):
        dict_action_space = {}
        for agent_id in range(self.num_agents):
            if self.observation_types[agent_id] not in ['mcts', 'full']:
                raise NotImplementedError
            else:
                obs_type = 'mcts'
            visible_graph = self.get_observation(agent_id, obs_type)
            dict_action_space[agent_id] = [node['id'] for node in visible_graph['nodes']]
        return dict_action_space

    def get_observation(self, agent_id, obs_type, info={}):

        if obs_type == 'mcts':
            return self.env.get_observations(char_index=agent_id)

        elif obs_type == 'full':
            return self.get_graph()

        else:
            pdb.set_trace()
            raise NotImplementedError
