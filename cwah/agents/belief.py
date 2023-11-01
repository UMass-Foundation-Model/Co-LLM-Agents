import numpy as np
import random
from simulation.evolving_graph.utils import load_graph_dict, load_name_equivalence
from simulation.evolving_graph.environment import EnvironmentState, EnvironmentGraph, GraphNode
import scipy.special
import ipdb
import pdb
import sys
import simulation.evolving_graph.utils as vh_utils
import json
import copy


class Belief():
    def __init__(self, graph_gt, agent_id, prior=None, forget_rate=0.0, seed=None, task_goal = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if task_goal is not None:
            self.goal_place = int([x for x in task_goal[0].keys()][0].split('_')[-1])
        else:
            self.goal_place = None
        # Possible beliefs for some objects
        self.container_restrictions = {
            'book': ['cabinet', 'kitchencabinet']
        }

        self.id_restrictions_inside = {
            
        }

        self.debug = False

        self.low_prob = -1e9
        self.name_equivalence = load_name_equivalence()
        self.map_properties_to_pred = {
            'ON': ('on', True),
            'OPEN': ('open', True),
            'OFF': ('on', False),
            'CLOSED': ('open', False)
        }

        self.map_edges_to_pred = {
            'INSIDE': 'inside',
            'CLOSE': 'close',
            'ON': 'ontop',
            'FACING': 'facing'
        }
        self.house_obj = [
            'floor',
            'wall',
            'ceiling'
        ]

        self.class_nodes_delete = ['wall', 'floor', 'ceiling', 'curtain', 'window']
        self.categories_delete = ['Doors']

        self.agent_id = agent_id
        self.grabbed_object = []
        # (prep, node_id, cur_node_max_belief[node_id]) eg : on, apple_id, table_id
        self.message_to_send = []
        self.node_max_belief = {}

        self.graph_helper = vh_utils.graph_dict_helper()
        self.binary_variables = self.graph_helper.binary_variables

        self.prohibit_ids = [node['id'] for node in graph_gt['nodes'] if
                             node['class_name'].lower() in self.class_nodes_delete or
                             node['category'] in self.categories_delete]
        new_graph = {
            'nodes': [copy.deepcopy(node) for node in graph_gt['nodes'] if node['id'] not in self.prohibit_ids],
            'edges': [edge for edge in graph_gt['edges'] if
                      edge['to_id'] not in self.prohibit_ids and edge['from_id'] not in self.prohibit_ids]
        }
        self.graph_init = graph_gt
        # ipdb.set_trace()
        self.sampled_graph = new_graph

        self.states_consider = ['OFF', 'CLOSED']
        self.edges_consider = ['INSIDE', 'ON']
        self.node_to_state_belief = {}
        self.room_node = {}
        self.room_nodes = []
        self.container_ids = []
        self.surface_ids = []

        # Binary Variable Dict
        self.bin_var_dict = {}
        for bin_var in self.binary_variables:
            self.bin_var_dict[bin_var.negative] = [[bin_var.positive, bin_var.negative], 0]
            self.bin_var_dict[bin_var.positive] = [[bin_var.positive, bin_var.negative], 1]

        id2node = {}
        for x in graph_gt['nodes']:
            id2node[x['id']] = x

        # Door edges: Will be used to make the graph walkable
        self.door_edges = {}
        for edge in self.graph_init['edges']:
            if edge['relation_type'] == 'BETWEEN':
                if id2node[edge['from_id']]['category'] == 'Doors':
                    if edge['from_id'] not in self.door_edges.keys():
                        self.door_edges[edge['from_id']] = []
                    self.door_edges[edge['from_id']].append(edge['to_id'])

        # Assume that between 2 nodes there is only one edge
        self.edge_belief = {}
        self.init_belief()

        self.record_edge_belief = copy.deepcopy(self.edge_belief)
        self.record_room_node = copy.deepcopy(self.room_node)

        self.first_belief = copy.deepcopy(self.edge_belief)
        self.first_room = copy.deepcopy(self.room_node)
        self.rate = forget_rate

    def update(self, origin, final):
        #DWH：currently self.rate is 0.0, so this function is useless.
        dist_total = origin - final
        ratio = (1 - np.exp(-self.rate * np.abs(origin - final)))
        return origin - ratio * dist_total
    
    def check_belief(self, belief):
        # deprecated.
        names = belief[0]
        probs = belief[1]
        max_prob_id = np.argmax(probs)
        if probs[max_prob_id] > 0.9:
            return names[max_prob_id], None
        not_in = []
        for i in range(len(probs)):
            if probs[i] <= self.low_prob:
                not_in.append(names[i])
        return None, not_in

    def delta_record_belief_old(self):
        # this function is used to check whether the belief has changed.
        # deprecated.
        need_to_report_inside = []
        need_to_report_not_inside = []
        for id in self.edge_belief.keys():
            # the only belief is INSIDE.
            sure_in_now, sure_not_in_now = self.check_belief(self.edge_belief[id]['INSIDE'])
            sure_in_old, sure_not_in_old = self.check_belief(self.record_edge_belief[id]['INSIDE'])
            if sure_in_now != sure_in_old and sure_in_now is not None:
                need_to_report_inside.append((id, sure_in_now))
            elif sure_not_in_now != sure_not_in_old and sure_not_in_now is not None:
                if sure_not_in_old is not None:
                    upd_not_in = [x for x in sure_not_in_now if x not in sure_not_in_old]
                else:
                    upd_not_in = sure_not_in_now
                need_to_report_not_inside.append((id, upd_not_in))
            room_sure_in_now, room_sure_not_in_now = self.check_belief(self.room_node[id])
            room_sure_in_old, room_sure_not_in_old = self.check_belief(self.record_room_node[id])
            if room_sure_in_now != room_sure_in_old and room_sure_in_now is not None:
                need_to_report_inside.append((id, room_sure_in_now))
            elif room_sure_not_in_now != room_sure_not_in_old and room_sure_not_in_now is not None:
                if room_sure_not_in_old is not None:
                    upd_not_in = [x for x in room_sure_not_in_now if x not in room_sure_not_in_old]
                else:
                    upd_not_in = room_sure_not_in_now
                need_to_report_not_inside.append((id, upd_not_in))
        return need_to_report_inside, need_to_report_not_inside

    def update_record_belief(self):
        self.record_edge_belief = copy.deepcopy(self.edge_belief)
        self.record_room_node = copy.deepcopy(self.room_node)

    def delta_record_belief_new(self):
        report = []
        for i, container_id in enumerate(self.container_ids): # 0 in room, and its id is None
            if container_id is None: continue
            upd = False
            for id in self.edge_belief.keys():
                if self.edge_belief[id]['INSIDE'][1][i] != self.record_edge_belief[id]['INSIDE'][1][i]:
                    # the container is checked
                    upd = True
            if upd:
                really_open = True
                obj_in_it = []
                for id in self.edge_belief.keys():
                    if self.edge_belief[id]['INSIDE'][1][i] > 0.9:
                        obj_in_it.append(id)
                    elif self.edge_belief[id]['INSIDE'][1][i] > self.low_prob:
                        really_open = False
                if really_open:
                    report.append((container_id, obj_in_it))

        for i, room_id in enumerate(self.room_ids):
            upd = False
            for id in self.edge_belief.keys():
                if self.room_node[id][1][i] != self.record_room_node[id][1][i] and self.room_node[id][1][i] > 0.9 and self.edge_belief[id]['INSIDE'][1][0] > 0.9:
                    # the room is checked
                    # Since the agent may only pass the room, we do not share 'NOT IN' info.
                    upd = True
            if upd:
                obj_in_it = []
                for id in self.edge_belief.keys():
                    if self.room_node[id][1][i] > 0.9:
                        obj_in_it.append(id)
                report.append((room_id, obj_in_it))
        return report
    
    def receive_belief_new(self, message):
        # message is a list of (node_id, [obj in it])
        for relation in message:
            if relation[0] in self.container_ids:
                for id in self.edge_belief.keys():
                    if id in relation[1]:
                        self.edge_belief[id]['INSIDE'][1][:] = self.low_prob
                        self.edge_belief[id]['INSIDE'][1][self.container_ids.index(relation[0])] = 1.0
                        self.record_edge_belief[id]['INSIDE'][1][:] = self.low_prob
                        self.record_edge_belief[id]['INSIDE'][1][self.container_ids.index(relation[0])] = 1.0
                    else:
                        if self.edge_belief[id]['INSIDE'][1][self.container_ids.index(relation[0])] < 0.9:
                            # If the agent is confident, he will not receive the message
                            # The case often happens, if a container is open, but the other agent does not see what is inside.
                            self.edge_belief[id]['INSIDE'][1][self.container_ids.index(relation[0])] = self.low_prob
                            self.record_edge_belief[id]['INSIDE'][1][self.container_ids.index(relation[0])] = self.low_prob
                        # Be careful, 0 is the room
            else:
                for id in self.room_node.keys():
                    if id in relation[1]:
                        self.room_node[id][1][:] = self.low_prob
                        self.room_node[id][1][self.room_ids.index(relation[0])] = 1.0
                        self.record_room_node[id][1][:] = self.low_prob
                        self.record_room_node[id][1][self.room_ids.index(relation[0])] = 1.0
                        self.edge_belief[id]['INSIDE'][1][:] = self.low_prob
                        self.edge_belief[id]['INSIDE'][1][0] = 1.0
                        self.record_edge_belief[id]['INSIDE'][1][:] = self.low_prob
                        self.record_edge_belief[id]['INSIDE'][1][0] = 1.0

    def receive_belief_old(self, message):
        # message is a list of (node_id, belief, type)
        # deprecated now
        for relation in message['INSIDE']:
            node_id = relation[0]
            if relation[1] in self.room_node[node_id][0]:
                container_pos = [i for i in range(len(self.room_node[node_id][0])) if self.room_node[node_id][0][i] in [relation[1]]]
                self.room_node[node_id][1][:] = self.low_prob
                self.record_room_node[node_id][1][:] = self.low_prob
                self.room_node[node_id][1][container_pos] = 1.0
                self.record_room_node[node_id][1][container_pos] = 1.0
            elif relation[1] in self.edge_belief[node_id]['INSIDE'][0]:
                container_pos = [i for i in range(len(self.edge_belief[node_id]['INSIDE'][0])) if self.edge_belief[node_id]['INSIDE'][0][i] in [relation[1]]]
                self.edge_belief[node_id]['INSIDE'][1][:] = self.low_prob
                self.record_edge_belief[node_id]['INSIDE'][1][:] = self.low_prob
                self.edge_belief[node_id]['INSIDE'][1][container_pos] = 1.0
                self.record_edge_belief[node_id]['INSIDE'][1][container_pos] = 1.0
            else:
                raise ValueError('Wrong type of relation')
        for relation in message['NOT_INSIDE']:
            node_id = relation[0]
            if len(relation[1]) == 0: continue # previous not inside, now in the room, from another agent, no need to update
            if relation[1][0] in self.room_node[node_id][0]:
                container_pos = [i for i in range(len(self.room_node[node_id][0])) if self.room_node[node_id][0][i] in relation[1]]
                self.room_node[node_id][1][container_pos] = self.low_prob
                self.record_room_node[node_id][1][container_pos] = self.low_prob
            elif relation[1][0] in self.edge_belief[node_id]['INSIDE'][0]:
                container_pos = [i for i in range(len(self.edge_belief[node_id]['INSIDE'][0])) if self.edge_belief[node_id]['INSIDE'][0][i] in relation[1]]
                self.edge_belief[node_id]['INSIDE'][1][container_pos] = self.low_prob
                self.record_edge_belief[node_id]['INSIDE'][1][container_pos] = self.low_prob
            else:
                raise ValueError('Wrong type of relation')

    def append_to_send(self, init=False):
        # first, calculate the max belief of each node now (only record max belief with high probability).
        # then, if init is False, for every nodes that is different from the max belief, append to the send message queue
        # finally, update the max belief of each node
        prep = 'INSIDE'
        cur_node_max_belief = {}
        def is_unique_max(arr, idx):
            return arr[idx] == arr.max() and np.count_nonzero(arr == arr.max()) == 1
        for node_id in self.edge_belief:
            names = self.edge_belief[node_id][prep][0]
            probs = self.edge_belief[node_id][prep][1]
            max_prob_id = np.argmax(probs)
            if is_unique_max(probs, max_prob_id):
                if names[max_prob_id] is None:
                    # check room
                    room_names = self.room_node[node_id][0]
                    room_probs = self.room_node[node_id][1]
                    max_room_prob_id = np.argmax(room_probs)
                    if is_unique_max(room_probs, max_room_prob_id):
                        cur_node_max_belief[node_id] = room_names[max_room_prob_id]
                        # self.message_to_send.append((prep, node_id, room_names[max_room_prob_id]))
                else:
                    cur_node_max_belief[node_id] = names[max_prob_id]
                    # self.message_to_send.append((prep, node_id, names[max_prob_id]))
        if not init:
            for node_id in cur_node_max_belief:
                if node_id not in self.node_max_belief or self.node_max_belief[node_id] != cur_node_max_belief[node_id]:
                    self.message_to_send.append((prep, node_id, cur_node_max_belief[node_id]))
        self.node_max_belief = cur_node_max_belief

    def receive_message(self, message):
        # message: (prep, node_id, state)
        # update the belief of the node_id
        if len(message) == 0:
            return
        message = eval(message)
        prep = message[0]
        node_id = message[1]
        container_or_room_id = message[2]
        if container_or_room_id in self.room_node[node_id][0]:
            self.room_node[node_id][1][:] = self.low_prob
            self.room_node[node_id][1][self.room_node[node_id][0].index(container_or_room_id)] = 1
            self.edge_belief[node_id][prep][1][:] = self.low_prob
            self.edge_belief[node_id][prep][1][self.edge_belief[node_id][prep][0].index(None)] = 1
        else:
            self.edge_belief[node_id][prep][1][:] = self.low_prob
            self.edge_belief[node_id][prep][1][self.edge_belief[node_id][prep][0].index(container_or_room_id)] = 1
        self.node_max_belief[node_id] = container_or_room_id # to prevent it send back again

    def update_to_prior(self):
        for node_name in self.edge_belief:
            self.edge_belief[node_name]['INSIDE'][1] = self.update(self.edge_belief[node_name]['INSIDE'][1],
                                                                   self.first_belief[node_name]['INSIDE'][1])

        for node in self.room_node:
            self.room_node[node][1] = self.update(self.room_node[node][1], self.first_room[node][1])

    def _remove_house_obj(self, state):
        delete_ids = [x['id'] for x in state['nodes'] if x['class_name'].lower() in self.class_nodes_delete]
        state['nodes'] = [x for x in state['nodes'] if x['id'] not in delete_ids]
        state['edges'] = [x for x in state['edges'] if x['from_id'] not in delete_ids and x['to_id'] not in delete_ids]
        return state

    def init_belief(self):
        # Set belief on object states, all states are binary and happen with a 50% chance
        for node in self.sampled_graph['nodes']:
            object_name = node['class_name']
            bin_vars = self.graph_helper.get_object_binary_variables(object_name)
            bin_vars = [x for x in bin_vars if x.default in self.states_consider]
            belief_dict = {}
            for bin_var in bin_vars:
                # 50% probablity of the state being positive (ON/OPEN) vs negative (OFF/CLOSED)
                belief_dict[bin_var.positive] = 0.5

            self.node_to_state_belief[node['id']] = belief_dict

        # TODO: ths class should simply have a surface property
        container_classes = [
            'bathroomcabinet',
            'kitchencabinet',
            'cabinet',
            'fridge',
            'stove',
            # 'kitchencounterdrawer',
            'dishwasher',
            # 'coffeepot',
            'microwave']

        # Solve these cases
        objects_inside_something = set(
            [edge['from_id'] for edge in self.sampled_graph['edges'] if edge['relation_type'] == 'INSIDE'])

        # Set belief for edges
        object_containers = [node for node in self.sampled_graph['nodes'] if node['class_name'] in container_classes]

        grabbable_nodes = [node for node in self.sampled_graph['nodes'] if 'GRABBABLE' in node['properties']]

        # TODO: this should be 0
        # grabbable_nodes = [node for node in grabbable_nodes if node['id'] in objects_inside_something]

        self.room_nodes = [node for node in self.sampled_graph['nodes'] if node['category'] == 'Rooms']

        self.room_ids = [x['id'] for x in self.room_nodes]
        container_ids = [x['id'] for x in object_containers]
        self.container_ids = [None] + container_ids
        self.room_index_belief_dict = {x: it for it, x in enumerate(self.room_ids)}

        self.container_index_belief_dict = {x: it for it, x in enumerate(self.container_ids) if x is not None}

        for obj_name in self.container_restrictions:
            possible_classes = self.container_restrictions[obj_name]
            # the ids that should not go here
            restricted_ids = [x['id'] for x in object_containers if x['class_name'] not in possible_classes]
            self.id_restrictions_inside[obj_name] = np.array(
                [self.container_index_belief_dict[rid] for rid in restricted_ids])

        """TODO: better initial belief"""
        object_room_ids = {}
        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE' and edge['to_id'] in self.room_ids:
                object_room_ids[edge['from_id']] = edge['to_id']

        nodes_inside_ids = [x['from_id'] for x in self.sampled_graph['edges'] if
                            x['to_id'] not in self.room_ids and x['relation_type'] == 'INSIDE']
        nodes_inside = [node for node in self.sampled_graph['nodes'] if
                        node['id'] in nodes_inside_ids and 'GRABBABLE' not in node['states']]

        objects_for_belief_reasoning = grabbable_nodes + nodes_inside
        # ipdb.set_trace()
        for node in objects_for_belief_reasoning:
            if node['class_name'] in self.class_nodes_delete or node['category'] in self.categories_delete:
                continue
            id1 = node['id']
            self.edge_belief[id1] = {}

            # Objects are inside containers with uniform probability, that is 1 / (1 + num_containers)
            # the 1 corresponds to the object not being inside any container
            init_values = np.ones(len(container_ids) + 1) / (1. + len(container_ids))

            # Special case for some object relationships that should not happen, just to encode some common sense
            if node['class_name'] in self.id_restrictions_inside.keys():
                init_values[self.id_restrictions_inside[node['class_name']]] = self.low_prob

            # The probability of being inside itself is 0
            if id1 in container_ids:
                init_values[1 + container_ids.index(id1)] = self.low_prob

            self.edge_belief[id1]['INSIDE'] = [[None] + container_ids, init_values]
            for i in range(len(self.edge_belief[id1]['INSIDE'][0])):
                if self.edge_belief[id1]['INSIDE'][0][i] is not None and self.edge_belief[id1]['INSIDE'][0][i] == self.goal_place:
                    self.edge_belief[id1]['INSIDE'][1][i] = self.low_prob

        # Room belief. Will be used for nodes that are not in the belief
        # Objects that are not inside anything, have probability 1/num_rooms to be in any of the rooms. We do not
        # divide here because we will be using a softmax later.
        for node in self.sampled_graph['nodes']:
            if node['class_name'] in self.class_nodes_delete or node['category'] in self.categories_delete:
                continue
            if node not in self.room_nodes:
                room_array = np.ones(len(self.room_ids)) / len(self.room_ids)
                self.room_node[node['id']] = [self.room_ids, room_array]
        self.sampled_graph['edges'] = []
        self.append_to_send(init=True)

    def reset_belief(self):
        self.sampled_graph['edges'] = []
        self.init_belief()

    def sample_from_belief(self, as_vh_state=False, ids_update=None):
        # Sample states
        for node in self.sampled_graph['nodes']:
            if ids_update is not None and node['id'] not in ids_update:
                continue
            if node['id'] not in self.node_to_state_belief:
                continue
            belief_node = self.node_to_state_belief[node['id']]
            states = []
            for var_name, var_belief_value in belief_node.items():
                rand_number = random.random()
                value_binary = 1 if rand_number < var_belief_value else 0
                states.append(self.bin_var_dict[var_name][0][value_binary])
            node['states'] = states

        node_inside = {}
        object_grabbed = []
        # Sample edges
        for edge in self.sampled_graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                node_inside[edge['from_id']] = edge['to_id']

            if edge['relation_type'] in ['HOLDS_LH', 'HOLDS_RH']:
                object_grabbed.append(edge['to_id'])

        for node in self.sampled_graph['nodes']:
            if ids_update is not None and node['id'] not in ids_update:
                continue
            if node['id'] not in self.edge_belief:
                if node['id'] not in self.room_node.keys():
                    continue
                node_room_cands = self.room_node[node['id']]
                node_room = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))
                final_rel = (node_room, 'INSIDE')
            else:
                edge_belief_inside = self.edge_belief[node['id']]['INSIDE']
                if node['id'] in node_inside:
                    # The relationships between unseen objects should stay the same
                    sample_inside = node_inside[node['id']]
                else:
                    try:
                        sample_inside = np.random.choice(edge_belief_inside[0],
                                                         p=scipy.special.softmax(edge_belief_inside[1]))
                    except:
                        print('Error with {}'.format(node['id']))
                        ipdb.set_trace()

                if sample_inside is None:
                    # Sample in a room
                    node_room_cands = self.room_node[node['id']]
                    node_room = np.random.choice(node_room_cands[0], p=scipy.special.softmax(node_room_cands[1]))
                    final_rel = (node_room, 'INSIDE')
                else:
                    if sample_inside == node['id']:
                        pass
                    final_rel = (sample_inside, 'INSIDE')

            if final_rel[1] == 'INSIDE':
                node_inside[node['id']] = final_rel[0]
            new_edge = {'from_id': node['id'], 'to_id': final_rel[0], 'relation_type': final_rel[1]}
            self.sampled_graph['edges'].append(new_edge)

        # try:
        #
        #     nodes_inside_graph = [edge['from_id'] for edge in self.sampled_graph['edges'] if
        #                           edge['relation_type'] == 'INSIDE']
        #     objects_grabbed = [edge['to_id'] for edge in self.sampled_graph['edges'] if
        #                        'HOLDS' in edge['relation_type']]
        #     nodes_inside_graph += objects_grabbed
        #     assert (len(set(self.edge_belief.keys()) - set(nodes_inside_graph)) == 0)
        # except:
        #     pdb.set_trace()

        # Include the doors
        for node_door in self.door_edges.keys():
            node_1, node_2 = self.door_edges[node_door]
            self.sampled_graph['edges'].append({'to_id': node_1, 'from_id': node_door, 'relation_type': 'BETWEEN'})
            self.sampled_graph['edges'].append({'to_id': node_2, 'from_id': node_door, 'relation_type': 'BETWEEN'})

        if as_vh_state:
            return self.to_vh_state(self.sampled_graph)

        return self.sampled_graph

    def to_vh_state(self, graph):
        state = self._remove_house_obj(graph)
        vh_state = EnvironmentState(EnvironmentGraph(state),
                                    self.name_equivalence, instance_selection=True)
        return vh_state

    def canopen_and_open(self, node):
        return 'CAN_OPEN' in node['properties'] and 'OPEN' in node['states']

    def is_surface(self, node):
        return 'SURFACE' in node['properties']

    def update_graph_from_gt_graph(self, gt_graph):
        """
        Updates the current sampled graph with a set of observations
        """
        # Here we have a graph sampled from our belief, and want to update it with gt graph
        id2node = {}
        gt_graph = {
            'nodes': [node for node in gt_graph['nodes'] if node['id'] not in self.prohibit_ids],
            'edges': [edge for edge in gt_graph['edges'] if
                      edge['from_id'] not in self.prohibit_ids and edge['to_id'] not in self.prohibit_ids]
        }

        edges_gt_graph = gt_graph['edges']
        for x in gt_graph['nodes']:
            id2node[x['id']] = x

        self.update_to_prior()
        self.update_from_gt_graph(gt_graph)

        char_node = self.agent_id

        inside = {}
        for x in edges_gt_graph:
            if x['relation_type'] == 'INSIDE':
                if x['from_id'] in inside.keys():
                    print('Already inside', id2node[x['from_id']]['class_name'],
                          id2node[inside[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])
                    raise Exception

                inside[x['from_id']] = x['to_id']

        for node in self.sampled_graph['nodes']:
            if node['id'] in id2node.keys():
                if ('states' not in id2node[node['id']].keys()):
                    print(node['id'], id2node[node['id']])
                states_graph_old = id2node[node['id']]['states']
                object_name = id2node[node['id']]['class_name']
                bin_vars = self.graph_helper.get_object_binary_variables(object_name)
                bin_vars = [x for x in bin_vars if x.default in self.states_consider]
                bin_vars_missing = [x for x in bin_vars if
                                    x.positive not in states_graph_old and x.negative not in states_graph_old]
                states_graph = states_graph_old + [x.default for x in bin_vars_missing]
                # fill out the rest of info regarding the states
                node['states'] = states_graph
                id2node[node['id']]['states'] = states_graph

        edges_keep = []
        ids_to_update = []
        for edge in self.sampled_graph['edges']:
            if edge['to_id'] in id2node.keys() and id2node[edge['to_id']]['category'] != 'Rooms' and inside[edge['to_id']] != inside[char_node] and edge['relation_type'] in ['INSIDE', 'ON'] and 'obs' in edge.keys():
                edges_keep.append(edge)
                continue
                #DWH add this.
                #The information comes from messages, and we just keep it
                #for example is (a) on (b) is known, and we got (c) on (b), we keep (a) on (b)
                #TODO: some bug in the code, when we see some object, some random edges have in it, and we should remove them.

            if (edge['from_id'] == char_node and edge['relation_type'] == 'INSIDE'):
                continue

            # Grabbed objects we don't need to keep them
            if edge['from_id'] == char_node and 'HOLD' in edge['relation_type']:
                continue

            # If the object should be visible but it is not in the observation, remove close relationship
            if (edge['from_id'] == char_node or edge['to_id'] == char_node) and edge['relation_type'] == 'CLOSE':
                continue

            # Objects that are visible, we do not care anymore
            if edge['from_id'] in id2node.keys() and edge['from_id'] != char_node:
                continue

            # The object is not visible but the container is visible

            if edge['to_id'] in id2node.keys() and edge['to_id'] != char_node:
                # If it is a room and we have not seen it, the belief remains
                if id2node[edge['to_id']]['category'] == 'Rooms' and edge['relation_type'] == 'INSIDE':
                    if inside[char_node] == edge['to_id']:
                        if edge['from_id'] not in id2node.keys():
                            ids_to_update.append(edge['from_id'])
                        else:
                            pass
                        continue
                else:
                    if edge['relation_type'] == 'ON':
                        ids_to_update.append(edge['from_id'])
                        continue
                    if edge['relation_type'] == 'INSIDE' and 'OPEN' in id2node[edge['to_id']]['states']:
                        ids_to_update.append(edge['from_id'])
                        continue
            edges_keep.append(edge)

        self.sampled_graph['edges'] = edges_keep + edges_gt_graph

        # For objects that are inside in the belief, character should also be close to those, so that when we open the object
        # we are already close to what is inside

        nodes_close = [x['to_id'] for x in edges_gt_graph if
                       x['from_id'] == char_node and x['relation_type'] == 'CLOSE']
        inside_belief = {}
        for edge in edges_keep:
            if edge['relation_type'] == 'INSIDE':
                if edge['to_id'] not in inside_belief: inside_belief[edge['to_id']] = []
                inside_belief[edge['to_id']].append(edge['from_id'])

        for node in nodes_close:
            if node not in inside_belief:
                continue

            for node_inside in inside_belief[node]:
                close_edges = [
                    {'from_id': char_node, 'to_id': node_inside, 'relation_type': 'CLOSE'},
                    {'to_id': char_node, 'from_id': node_inside, 'relation_type': 'CLOSE'}
                ]
                self.sampled_graph['edges'] += close_edges

        # sample new edges that have not been seen
        self.sample_from_belief(ids_update=ids_to_update)

        return self.sampled_graph

    def update_from_gt_graph(self, gt_graph):
        # Update the states of nodes that we can see in the belief. Note that this does not change the sampled graph
        id2node = {}
        for x in gt_graph['nodes']:
            id2node[x['id']] = x

        inside = {}

        grabbed_object = []
        for x in gt_graph['edges']:
            if x['relation_type'] in ['HOLDS_LH', 'HOLDS_RH']:
                grabbed_object.append(x['to_id'])

            if x['relation_type'] == 'INSIDE':
                if x['from_id'] in inside.keys():
                    print('Already inside', id2node[x['from_id']]['class_name'],
                          id2node[inside[x['from_id']]]['class_name'], id2node[x['to_id']]['class_name'])

                    raise Exception

                inside[x['from_id']] = x['to_id']

        visible_ids = [x['id'] for x in gt_graph['nodes']]
        edge_tuples = [(x['from_id'], x['to_id']) for x in gt_graph['edges']]

        for x in gt_graph['nodes']:
            try:
                dict_state = self.node_to_state_belief[x['id']]
                for state in x['states']:
                    pred_name = self.bin_var_dict[state][0][0]
                    dict_state[pred_name] = self.bin_var_dict[state][1]
            except:
                pass

        char_node = self.agent_id

        visible_room = inside[char_node]

        deleted_edges = []
        id_updated = []

        # Keep track of things with impossible belief
        # Objects and rooms we are just seeing
        ids_known_info = [self.room_index_belief_dict[visible_room], []]
        for id_node in self.edge_belief.keys():
            id_updated.append(id_node)
            if id_node in grabbed_object:
                continue

            if id_node in visible_ids:
                # TODO: what happens when object grabbed
                assert (id_node in inside.keys())
                inside_obj = inside[id_node]

                # Some objects have the relationship inside but they are not part of the belief because
                # they are visible anyways like bookshelf. In that case we consider them to just be
                # inside the room
                if inside_obj not in self.room_ids and inside_obj not in self.container_index_belief_dict:
                    inside_obj = inside[inside_obj]

                # If object is inside a room, for sure it is not insde another object
                if inside_obj in self.room_ids:
                    self.edge_belief[id_node]['INSIDE'][1][:] = self.low_prob
                    self.edge_belief[id_node]['INSIDE'][1][0] = 1.
                    self.room_node[id_node][1][:] = self.low_prob
                    self.room_node[id_node][1][self.room_index_belief_dict[inside_obj]] = 1.
                else:
                    # If object is inside an object, for sure it is not insde another object
                    index_inside = self.container_index_belief_dict[inside_obj]
                    self.edge_belief[id_node]['INSIDE'][1][:] = self.low_prob
                    self.edge_belief[id_node]['INSIDE'][1][index_inside] = 1.


            else:
                # If not visible. for sure not in this room
                self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = self.low_prob
                if (self.room_node[id_node][1] > self.low_prob).sum() == 0:
                    if id_node in self.edge_belief:
                        # If not in any room, needs to be inside something
                        self.edge_belief[id_node]['INSIDE'][1][0] = self.low_prob

            for id_node in self.container_ids:
                if id_node in visible_ids and id_node in self.container_ids and 'OPEN' in id2node[id_node]['states']:
                    for id_node_child in self.edge_belief.keys():
                        if id_node_child not in inside.keys() or inside[id_node_child] != id_node:
                            ids_known_info[1].append(self.container_index_belief_dict[id_node])
                            self.edge_belief[id_node_child]['INSIDE'][1][
                                self.container_index_belief_dict[id_node]] = self.low_prob

        # Some furniture has no edges, only has info about inside rooms
        # We need to udpate its location
        for id_node in self.room_node.keys():
            if id_node not in id_updated:
                if id_node in visible_ids:
                    inside_obj = inside[id_node]
                    if inside_obj == visible_room:
                        self.room_node[id_node][1][:] = self.low_prob
                        self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = 1.
                    else:
                        assert ('Error: A grabbable object is inside something else than a room')
                else:
                    # Either the node goes inside somehting in the room... or ti should not be
                    # in this room
                    self.room_node[id_node][1][self.room_index_belief_dict[visible_room]] = self.low_prob

        mask_house = np.ones(len(self.room_nodes))
        mask_obj = np.ones(len(self.container_ids))
        mask_house[ids_known_info[0]] = 0

        assert (len(self.room_nodes) > 0)
        if len(ids_known_info[1]):
            mask_obj[np.array(ids_known_info[1])] = 0

        mask_obj = (mask_obj == 1)
        mask_house = (mask_house == 1)
        # Check for impossible beliefs
        for id_node in self.room_node.keys():
            if id_node in self.edge_belief.keys():
                # the object should be in a room or inside something
                if np.max(self.edge_belief[id_node]['INSIDE'][1]) == self.low_prob:
                    # Sample locations except for marked ones
                    self.edge_belief[id_node]['INSIDE'][1] = self.first_belief[id_node]['INSIDE'][1]
                    # Sample rooms except marked
                    try:
                        self.room_node[id_node][1][mask_house] = self.first_room[id_node][1][mask_house]
                    except:
                        pdb.set_trace()
            else:
                if np.max(self.room_node[id_node][1]) == self.low_prob:
                    self.room_node[id_node][1][mask_house] = self.first_room[id_node][1][mask_house]


if __name__ == '__main__':
    graph_init = '../../example_graph/example_graph.json'
    with open(graph_init, 'r') as f:
        graph = json.load(f)['init_graph']
    Belief(graph)
