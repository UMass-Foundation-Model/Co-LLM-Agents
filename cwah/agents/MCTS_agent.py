import numpy as np
import random
import time
import math
import copy
import importlib
import json
import multiprocessing
import ipdb
import pickle

from . import belief
from . import utils
from envs.graph_env import VhGraphEnv
#
from MCTS import *

import sys

sys.path.append('..')
from utils import utils_environment as utils_env


def find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node['id']: node for node in env_graph['nodes']}
    containerdict = {edge['from_id']: edge['to_id'] for edge in env_graph['edges'] if edge['relation_type'] == 'INSIDE'}
    target = int(object_target.split('_')[-1])
    observation_ids = [x['id'] for x in observations['nodes']]
    try:
        room_char = [edge['to_id'] for edge in env_graph['edges'] if
                     edge['from_id'] == agent_id and edge['relation_type'] == 'INSIDE'][0]
    except:
        print('Error')
        # ipdb.set_trace()

    action_list = []
    cost_list = []
    # if target == 478:
    #     ipdb.set_trace()
    count = 0
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            print(observation_ids)
            print(observations)
            print(env_graph)
            ipdb.set_trace()
        # If the object is a room, we have to walk to what is inside

        if id2node[container]['category'] == 'Rooms':
            action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list
            cost_list = [0.5] + cost_list

        elif 'CLOSED' in id2node[container]['states'] or ('OPEN' not in id2node[container]['states']):
            action = ('open', (id2node[container]['class_name'], container), None)
            action_list = [action] + action_list
            cost_list = [0.05] + cost_list

        if (count > 5): print(target)
        count += 1
        target = container

    ids_character = [x['to_id'] for x in observations['edges'] if
                     x['from_id'] == agent_id and x['relation_type'] == 'CLOSE'] + \
                    [x['from_id'] for x in observations['edges'] if
                     x['to_id'] == agent_id and x['relation_type'] == 'CLOSE']

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [('walk', (id2node[target]['class_name'], target), None)] + action_list
        cost_list = [1] + cost_list

    return action_list, cost_list


def grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (
                (edge['from_id'] == agent_id and edge['to_id'] == target_id) or (
                    edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if
                       (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('grab', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator,
                                                  object_target)
        return find_actions + target_action, find_costs + cost


def turnOn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (
                (edge['from_id'] == agent_id and edge['to_id'] == target_id) or (
                    edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    grabbed_obj_ids = [edge['to_id'] for edge in env_graph['edges'] if
                       (edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [('switchon', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator,
                                                  object_target)
        return find_actions + target_action, find_costs + cost


def sit_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, object_target):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split('_')[-1])

    observed_ids = [node['id'] for node in observations['nodes']]
    agent_close = [edge for edge in env_graph['edges'] if (
                (edge['from_id'] == agent_id and edge['to_id'] == target_id) or (
                    edge['from_id'] == target_id and edge['to_id'] == agent_id) and edge['relation_type'] == 'CLOSE')]
    on_ids = [edge['to_id'] for edge in env_graph['edges'] if
              (edge['from_id'] == agent_id and 'ON' in edge['relation_type'])]

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_id][0]

    if target_id not in on_ids:
        target_action = [('sit', (target_node['class_name'], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost
    else:
        find_actions, find_costs = find_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator,
                                                  object_target)
        return find_actions + target_action, find_costs + cost


def put_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if
            edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        # Object has been placed
        print(f"bad observations in put_heuristic with target_grap {target_grab} and target_put {target_put}")
        return [], []

    if sum([1 for edge in observations['edges'] if
            edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if
                          edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge[
                              'to_id'] == target_grab]) > 0

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator,
                                                   'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room

        env_graph_new = copy.deepcopy(env_graph)

        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})

        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge[
                                          'relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, unsatisfied, env_graph_new, simulator,
                                               'find_' + str(target_node2['id']))
    action = [('putback', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost = [0.05]
    res = grab_obj1 + find_obj2 + action
    cost_list = cost_grab_obj1 + cost_find_obj2 + cost

    # print(res, target)
    return res, cost_list


def putIn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split('_')[-2:]]

    if sum([1 for edge in observations['edges'] if
            edge['from_id'] == target_grab and edge['to_id'] == target_put and edge['relation_type'] == 'ON']) > 0:
        #TODO: dwh marked it as a bug in original mcts code. Is the relationship type correct?
        print(f"bad observations in putIn_heuristic with target_grap {target_grab} and target_put {target_put}")
        return [], []

    if sum([1 for edge in observations['edges'] if
            edge['to_id'] == target_grab and edge['from_id'] != agent_id and 'HOLD' in edge['relation_type']]) > 0:
        # Object has been placed
        return None, None

    target_node = [node for node in env_graph['nodes'] if node['id'] == target_grab][0]
    target_node2 = [node for node in env_graph['nodes'] if node['id'] == target_put][0]
    id2node = {node['id']: node for node in env_graph['nodes']}
    target_grabbed = len([edge for edge in env_graph['edges'] if
                          edge['from_id'] == agent_id and 'HOLDS' in edge['relation_type'] and edge[
                              'to_id'] == target_grab]) > 0

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1 = grab_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator,
                                                   'grab_' + str(target_node['id']))
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == 'walk':
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]['category'] == 'Rooms':
                    object_diff_room = id_room

        env_graph_new = copy.deepcopy(env_graph)

        if object_diff_room:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      edge['to_id'] != agent_id and edge['from_id'] != agent_id]
            env_graph_new['edges'].append({'from_id': agent_id, 'to_id': object_diff_room, 'relation_type': 'INSIDE'})

        else:
            env_graph_new['edges'] = [edge for edge in env_graph_new['edges'] if
                                      (edge['to_id'] != agent_id and edge['from_id'] != agent_id) or edge[
                                          'relation_type'] == 'INSIDE']
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
    find_obj2, cost_find_obj2 = find_heuristic(agent_id, char_index, unsatisfied, env_graph_new, simulator,
                                               'find_' + str(target_node2['id']))
    target_put_state = target_node2['states']
    action_open = [('open', (target_node2['class_name'], target_put))]
    action_put = [('putin', (target_node['class_name'], target_grab), (target_node2['class_name'], target_put))]
    cost_open = [0.05]
    cost_put = [0.05]

    remained_to_put = 0
    for predicate, count in unsatisfied.items():
        if predicate.startswith('inside'):
            remained_to_put += count
    if remained_to_put == 1:  # or agent_id > 1:
        action_close = []
        cost_close = []
    else:
        action_close = [('close', (target_node2['class_name'], target_put))]
        cost_close = [0.05]

    if 'CLOSED' in target_put_state or 'OPEN' not in target_put_state:
        res = grab_obj1 + find_obj2 + action_open + action_put + action_close
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put + cost_close
    else:
        res = grab_obj1 + find_obj2 + action_put + action_close
        cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put + cost_close

    # print(res, target)
    return res, cost_list


def clean_graph(state, goal_spec, last_opened):
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate in goal_spec:
        elements = predicate.split('_')
        nodes_missing += [int(x) for x in elements if x.isdigit()]
        for x in elements[1:]:
            if x.isdigit():
                nodes_missing += [int(x)]
            else:
                nodes_missing += [node['id'] for node in state['nodes'] if node['class_name'] == x]
    nodes_missing += [node['id'] for node in state['nodes'] if
                      node['class_name'] == 'character' or node['category'] in ['Rooms', 'Doors']]

    id2node = {node['id']: node for node in state['nodes']}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state['edges']:
        if edge['relation_type'] == 'INSIDE':
            if edge['from_id'] not in inside.keys():
                inside[edge['from_id']] = []
            inside[edge['from_id']].append(edge['to_id'])

    while (len(nodes_missing) > 0):
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [node_in for node_in in inside[node_missing] if node_in not in ids_interaction]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['dishwasher', 'kitchentable']:
                augmented_class_names += ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']
                break
    for key in goal_spec:
        elements = key.split('_')
        if elements[0] == 'off':
            if id2node[int(elements[2])]['class_name'] in ['sofa', 'chair']:
                augmented_class_names += ['coffeetable']
                break
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if
                  node['class_name'] in augmented_class_names]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    new_graph = {
        "edges": [edge for edge in state['edges'] if
                  edge['from_id'] in ids_interaction and edge['to_id'] in ids_interaction],
        "nodes": [id2node[id_node] for id_node in ids_interaction]
    }

    return new_graph


def get_plan(sample_id, root_action, root_node, env, mcts, nb_steps, goal_spec, res, last_subgoal, last_action,
             opponent_subgoal=None, verbose=True):
    init_state = env.state

    if True:  # clean graph
        init_state = clean_graph(init_state, goal_spec, mcts.last_opened)
        init_vh_state = env.get_vh_state(init_state)
    else:
        init_vh_state = env.vh_state

    satisfied, unsatisfied = utils_env.check_progress(init_state, goal_spec)
    remained_to_put = 0
    for predicate, count in unsatisfied.items():
        if predicate.startswith('inside'):
            remained_to_put += count

    if last_action is not None and last_action.split(' ')[
        0] == '[putin]' and remained_to_put == 0:  # close the door (may also need to check if it has a door)
        elements = last_action.split(' ')
        action = '[close] {} {}'.format(elements[3], elements[4])
        plan = [action]
        subgoals = [last_subgoal]

    # if root_action is None:
    root_node = Node(id=(root_action, [init_vh_state, init_state, goal_spec, satisfied, unsatisfied, 0, []]),
                     num_visited=0,
                     sum_value=0,
                     is_expanded=False)
    curr_node = root_node
    heuristic_dict = {
        'find': find_heuristic,
        'grab': grab_heuristic,
        'put': put_heuristic,
        'putIn': putIn_heuristic,
        'sit': sit_heuristic,
        'turnOn': turnOn_heuristic
    }
    next_root, plan, subgoals = mcts.run(curr_node,
                                         nb_steps,
                                         heuristic_dict,
                                         last_subgoal,
                                         opponent_subgoal)
    if verbose:
        print('plan', plan)
        print('subgoal', subgoals)
    if sample_id is not None:
        res[sample_id] = plan
    else:
        return plan, next_root, subgoals


class MCTS_agent:
    """
    MCTS for a single agent
    """

    def __init__(self, agent_id, char_index,
                 max_episode_length, num_simulation, max_rollout_steps, c_init, c_base, recursive=False,
                 num_samples=1, num_processes=1, comm=None, logging=False, logging_graphs=False, seed=None,
                 belief_comm=False, opponent_subgoal='None', satisfied_comm=False):
        self.agent_type = 'MCTS'
        self.verbose = False
        self.recursive = recursive
        # self.env = unity_env.env
        if seed is None:
            seed = random.randint(0, 100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs
        self.with_character_id = [agent_id]
        self.agent_id = agent_id
        self.char_index = char_index
        self.sim_env = VhGraphEnv()
        self.sim_env.pomdp = True
        self.belief = None
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        self.previous_subgoal = None
        self.belief_comm = belief_comm
        self.opponent_subgoal = opponent_subgoal
        self.satisfied_comm = satisfied_comm
        self.received_opponent_subgoal = None
        self.step = 0
        self.previous_belief_graph = None
        self.verbose = False
        self.previous_room = None
        self.mcts = MCTS(self.sim_env, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base)

        if self.mcts is None:
            raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm

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
        self.belief.append_to_send()
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    # def filter_msg_with_target_id(self, goal_ids):
    #     self.belief.message_to_send = [msg for msg in self.belief.message_to_send if
    #                                    msg[1] in goal_ids and msg[1] not in self.belief.grabbed_object]
    def filter_msg_with_target_class(self, goal_classes, id_to_class):
        try:
            self.belief.message_to_send = [msg for msg in self.belief.message_to_send if id_to_class[msg[1]] in goal_classes]
        except KeyError:
            print('----------------------------------------------')
            print(self.belief.message_to_send)
            print(id_to_class)
            

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
        edges = [edge for edge in graph['edges'] if edge['from_id'] == char_id]
        print('Character:')
        print(edges)
        print('---')

    def get_action(self, obs, goal_spec, opponent_subgoal=None):
        if self.opponent_subgoal == 'None':
            opponent_subgoal = None
        msg_to_send = ""
        if obs['messages'] != None and len(obs['messages']) > 1 and obs['messages'][1 - self.char_index]:
            if self.char_index == 0 and obs['messages'][0] != None: msg = {} # do not recurrent
            else: msg = eval(utils.language_to_MCTS_convert(obs['messages'][1 - self.char_index]))
            # check message state
            if self.opponent_subgoal == 'comm':
                if 'S' in msg.keys():
                    self.received_opponent_subgoal = msg['S']
                    if self.received_opponent_subgoal is not None:
                        opponent_subgoal = self.received_opponent_subgoal[:-1]
                    else:
                        opponent_subgoal = None
            if self.belief_comm:
                if 'B' in msg.keys():
                    self.belief.receive_belief_new(msg['B'])
            if self.satisfied_comm:
                if 'E' in msg.keys():
                    # 'edges': [{'from_id': 57, 'to_id': 56, 'relation_type': 'INSIDE'}, ...
                    def add_distinct(l1, l2):
                        for i in l2:
                            for j in l1:
                                if i == j:
                                    break
                            else:
                                l1.append(i)
                    add_relation = {}
                    if msg['E']['relation_type'] == 'INSIDE':
                        add_relation['edges'] = [{'from_id': msg['E']['from_id'], 'to_id': msg['E']['to_id'], 'relation_type': 'INSIDE'},
                                                    {'from_id': msg['E']['to_id'], 'to_id': msg['E']['room_id'], 'relation_type': 'INSIDE'}]
                    elif msg['E']['relation_type'] == 'ON':
                        add_relation['edges'] = [{'from_id': msg['E']['from_id'], 'to_id': msg['E']['to_id'], 'relation_type': 'ON'},
                                                    {'from_id': msg['E']['to_id'], 'to_id': msg['E']['room_id'], 'relation_type': 'INSIDE'},
                                                    {'from_id': msg['E']['from_id'], 'to_id': msg['E']['room_id'], 'relation_type': 'INSIDE'}]
                    relative_id = list(set([x['from_id'] for x in add_relation['edges']] + [x['to_id'] for x in add_relation['edges']]))
                    add_relation['nodes'] = [
                        [{'class_name': y['class_name'],
                        'id': y['id'],
                        'category': y['category'],
                        'properties': y['properties'],
                        'states': y['states']}
                        for y in self.sim_env.state['nodes'] if y['id'] == x][0] for x in relative_id]
                    add_distinct(obs['nodes'], add_relation['nodes'])
                    add_distinct(obs['edges'], add_relation['edges'])

        obs['edges'] = [{**x, 'obs': True} for x in obs['edges']]
        self.sample_belief(obs)
        self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        last_action = self.last_action
        last_subgoal = self.last_subgoal

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None
        verbose = self.verbose

        plan, root_node, subgoals = get_plan(None, root_action, root_node, self.sim_env, self.mcts, nb_steps, goal_spec,
                                             None, last_subgoal, last_action, opponent_subgoal, verbose=verbose)
        if self.opponent_subgoal == 'comm' and self.received_opponent_subgoal is not None and self.received_opponent_subgoal[-1] == '0': self.received_opponent_subgoal = None
        
        # ipdb.set_trace()
        if len(plan) > 0:
            action = plan[0]
            action = action.replace('[walk]', '[walktowards]')
        else:
            action = None
        self.last_subgoal = subgoals[0] if len(subgoals) > 0 else  None
        if self.logging:
            info = {
                'plan': plan[:3] if len(plan) > 3 else plan,
                'subgoals': subgoals,
                'subgoal': self.last_subgoal,
                'belief': copy.deepcopy(self.belief.edge_belief),
                'belief_graph': copy.deepcopy(self.sim_env.vh_state.to_dict())
            }
            if self.logging_graphs:
                info.update(
                    {'obs': obs['nodes']})
        else:
            info = {}
        
        roomname = [edge['to_id'] for edge in self.sim_env.state['edges'] if edge['from_id'] == self.agent_id and edge['relation_type'] == 'INSIDE'][0]
        
        if self.satisfied_comm:
            # if previous action is putback or putin, then send this info to other. 
            last_action_name = utils_env.get_action_name(last_action)
            if last_action_name == "putback" or last_action_name == "putin":
                roomname = [edge['to_id'] for edge in self.sim_env.state['edges'] if edge['from_id'] == self.agent_id and edge['relation_type'] == 'INSIDE'][0]
                objectname = utils_env.get_id(last_action)
                containername = utils_env.get_last_id(last_action)
                if last_action_name == "putback":
                    msg_to_send_edges = f"{{'from_id': {objectname}, 'to_id': {containername}, 'relation_type': 'ON', 'room_id': {roomname}}}"
                if last_action_name == "putin":
                    msg_to_send_edges = f"{{'from_id': {objectname}, 'to_id': {containername}, 'relation_type': 'INSIDE', 'room_id': {roomname}}}"
                #see whether we need to send node info
                msg_to_send += f"'E':{msg_to_send_edges},"

        # start to decide whether to send message
        decide_send_this_turn = (self.step >= self.char_index) # avoid stuck
        # random_decide_send_this_turn = random.sample([True, False], 1)[0]
        if self.opponent_subgoal == 'comm':
            if self.previous_subgoal != self.last_subgoal:
                if self.last_subgoal is not None:
                    if decide_send_this_turn:
                        obj = int(self.last_subgoal.split('_')[1])
                        if obj in [i['id'] for i in obs['nodes']]:
                            msg_to_send += f"'S':'{str(self.last_subgoal) + '_1'}',"
                        else:
                            msg_to_send += f"'S':'{str(self.last_subgoal) + '_0'}',"
        if decide_send_this_turn:
            self.previous_subgoal = self.last_subgoal
                        # action = f'[send_message] <S{str(self.last_subgoal)}>'
        send_belief_flag = False
        id_to_class = {self.sim_env.state['nodes'][i]['id']: self.sim_env.state['nodes'][i]['class_name'] for i in range(len(self.sim_env.state['nodes']))}
        if self.belief_comm and roomname != self.previous_room:
            upd_belief_info = self.belief.delta_record_belief_new()
            ralative_object = [i.split('_')[1] for i in goal_spec.keys()]
            filter_id_list = lambda x: [i for i in x if id_to_class[i] in ralative_object]
            upd_belief_info = [(x[0], filter_id_list(x[1])) for x in upd_belief_info]
            total_value = 0
            filter_upd_info = []
            for one_info in upd_belief_info:
                filter_one_info = []
                for obj in one_info[1]:
                    if obj not in self.with_character_id:
                        filter_one_info.append(obj)
                        total_value += 4
                filter_upd_info.append((one_info[0], filter_one_info))
                total_value += 1
            if total_value >= 5: # Balance between communication cost and information gain
                msg_to_send += f"'B':{filter_upd_info},"
                send_belief_flag = True
                self.belief.update_record_belief()
        # self.id_to_class.update({node['id']: node['class_name'] for node in obs['nodes']})
        if msg_to_send != "":
            msg_to_send = '{' + msg_to_send + '}'
            text_message = utils.MCTS_to_language_convert(msg_to_send, id_to_name_api = lambda x: id_to_class[x])
            action = f'[send_message] <{text_message}>'
        self.step += 1
        self.previous_room = roomname
    #    if obs['location'] == self.last_position and 'walk' in self.last_action:
    #        assert -1 == 1, "stuck"
        self.last_action = action
    #    print(self.last_position)
        self.last_position = obs['location']
        actionname = utils_env.get_action_name(action)
        firstid = utils_env.get_id(action)
        if 'grab' == actionname:
            self.with_character_id.append(firstid)
        if 'put' == actionname or 'putin' == actionname or 'putback' == actionname:
            self.with_character_id.remove(firstid)
        return action, info

    def reset(self, observed_graph, gt_graph, task_goal, seed=0, simulator_type='python', is_alice=False):
        self.step = 0
        self.last_action = None
        self.last_subgoal = None
        self.received_opponent_subgoal = None
        self.last_position = None
        self.with_character_id = [self.agent_id]
        """TODO: do no need this?"""
        self.previous_room = None
        self.previous_belief_graph = None
        self.belief = belief.Belief(gt_graph, agent_id=self.agent_id, seed=seed)
        # print("set")
        self.belief.sample_from_belief()
        graph_belief = self.sample_belief(observed_graph)  # self.env.get_observations(char_index=self.char_index))
        try:
            self.sim_env.reset(graph_belief, task_goal)
        except:
            import ipdb

            ipdb.set_trace()
        self.sim_env.to_pomdp()
        self.mcts = MCTS(self.sim_env, self.agent_id, self.char_index, self.max_episode_length,
                         self.num_simulation, self.max_rollout_steps,
                         self.c_init, self.c_base, seed=seed)

