import random
import numpy as np
from anytree import AnyNode as Node
import copy
import ipdb
from tqdm import tqdm

class MCTS:
    def __init__(self, env, agent_id, char_index, max_episode_length, num_simulation, max_rollout_step, c_init, c_base, seed=1, verbose = False):
        self.env = env
        self.discount = 1.0#0.4
        self.agent_id = agent_id
        self.char_index = char_index
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_step = max_rollout_step
        self.c_init = c_init 
        self.c_base = c_base
        self.seed = 1
        self.heuristic_dict = None
        self.opponent_subgoal = None
        self.last_opened = None
        self.verbose = verbose
        np.random.seed(self.seed)
        random.seed(self.seed)


    def check_progress(self, state, goal_spec):
        """TODO: add more predicate checkers; currently only ON"""
        count = 0
        for key, value in goal_spec.items():
            if key.startswith('off'):
                count += value
        id2node = {node['id']: node for node in state['nodes']}
        for key, value in goal_spec.items():
            elements = key.split('_')
            for edge in state['edges']:
                if elements[0] in ['on', 'inside']:
                    if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                        count += 1
                elif elements[0] == 'offOn':
                    if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                        count -= 1
                elif elements[1] == 'offInside':
                    if edge['relation_type'].lower() == 'inside' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                        count -= 1
                elif elements[0] == 'holds':
                    if edge['relation_type'].lower().startswith('holds') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                        count += 1
                elif elements[0] == 'sit':
                    if edge['relation_type'].lower().startswith('on') and edge['to_id'] == int(elements[2]) and edge['from_id'] == int(elements[1]):
                        count += 1
            if elements[0] == 'turnOn':
                if 'ON' in id2node[int(elements[1])]['states']:
                    count += 1
        return count
        
    def run(self, curr_root, t, heuristic_dict, last_subgoal, opponent_subgoal):
        self.opponent_subgoal = opponent_subgoal
        if self.verbose:
            print('check subgoal')
        curr_vh_state_tmp, curr_state_tmp, _, satisfied, unsatisfied, _, actions_parent = curr_root.id[1]
        subgoals = self.get_subgoal_space(curr_state_tmp, satisfied, unsatisfied, opponent_subgoal, verbose=1)
        # subgoals = [sg for sg in subgoals if sg[0] != opponent_subgoal] # avoid repreating
        #print(curr_state_tmp['edges'])

        if self.verbose:
            print('satisfied:', satisfied)
            print('unsatisfied:', unsatisfied)
            print('subgoals:', subgoals)
            print('last_subgoal:', last_subgoal)

        """TODO: check predicates other than ON"""
        id2node = {node['id']: node for node in curr_state_tmp['nodes']}
        inhand_objs = [id2node[edge['to_id']]['class_name'] for edge in curr_state_tmp['edges'] if edge['relation_type'].startswith('HOLDS') \
                and self.agent_id == edge['from_id']]
        needed_obj_count = {}
        # needed_container = []
        for predicate, count in unsatisfied.items():
            elements = predicate.split('_')
            if elements[0] in ['on', 'inside']:
                # if elements[0] == 'inside':
                #     needed_container.append(int(elements[2]))
                if elements[1] not in needed_obj_count:
                    needed_obj_count[elements[1]] = count
                else:
                    needed_obj_count[elements[1]] += count
                if elements[1] in inhand_objs:
                    needed_obj_count[elements[1]] -= 1

        remained_to_put = {}
        for predicate, count in unsatisfied.items():
            elements = predicate.split('_')
            if elements[0] == 'inside':
                if int(elements[2]) not in remained_to_put:
                    remained_to_put[int(elements[2])] = count
                else:
                    remained_to_put[int(elements[2])] += count

        need_to_close = False
        if self.last_opened is None: # add close & opened containers
            for edge in curr_state_tmp['edges']:
                if edge['from_id'] == self.agent_id and edge['relation_type'] == 'CLOSE' and \
                    id2node[edge['to_id']]['class_name'] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave', 'dishwasher', 'stove'] and \
                    'OPEN' in id2node[edge['to_id']]['states']:
                    self.last_opened = ['<{}>'.format(id2node[edge['to_id']]['class_name']), '({})'.format(edge['to_id'])]
                    
        if self.last_opened is not None and self.last_opened[0] != '<toilet>':
            for node in curr_state_tmp['nodes']:
                # print(node)
                # ipdb.set_trace()
                if '({})'.format(node['id']) == self.last_opened[1]:
                    # print('check opened node:', node)
                    # ipdb.set_trace()
                    if 'OPEN' in node['states']:
                        # if self.agent_id == 2:# and node['id'] == 323:
                        #     ipdb.set_trace()
                        if node['id'] in remained_to_put:
                            need_to_close = remained_to_put[node['id']] == 0
                        else:
                            need_to_close = True
                            
                            inside_objs = [edge['from_id'] for edge in curr_state_tmp['edges'] if edge['relation_type'] == 'INSIDE' and '({})'.format(edge['to_id']) == self.last_opened[1]]
                            for obj_id in inside_objs:
                                if str(obj_id) in needed_obj_count and needed_obj_count[str(obj_id)] > 0 or \
                                   id2node[obj_id]['class_name'] in needed_obj_count and needed_obj_count[id2node[obj_id]['class_name']] > 0:
                                   need_to_close = False
                                   break
                    else:
                        self.last_opened = None
                    break
        need_to_close = False
        if self.verbose:
            print('last_opened:', self.last_opened, need_to_close)
        # if self.agent_id > 1:
        #     need_to_close = False
        for subgoal in subgoals:
            if subgoal[0] == last_subgoal:
                heuristic = heuristic_dict[last_subgoal.split('_')[0]]
                actions, costs = heuristic(self.agent_id, self.char_index, unsatisfied, curr_state_tmp, self.env, last_subgoal)
                if actions is None:
                    plan = []
                else:
                    plan = [self.get_action_str(action) for action in actions]
                if len(plan) > 0:
                    elements = plan[0].split(' ')               
                    if need_to_close and (elements[0] == '[walk]' or elements[0] == '[open]' and elements[2] != self.last_opened[1]):
                        if self.last_opened is not None:
                            for edge in curr_state_tmp['edges']:
                                if edge['relation_type'] == 'CLOSE' and \
                                    ('({})'.format(edge['from_id']) == self.last_opened[1] and edge['to_id'] == self.agent_id or \
                                    '({})'.format(edge['to_id']) == self.last_opened[1] and edge['from_id'] == self.agent_id):
                                    plan = ['[close] {} {}'.format(self.last_opened[0], self.last_opened[1])] + plan
                                    break
                if self.verbose:
                    print('repeat subgoal plan:', plan)
                if len(plan) > 0 and plan[0].startswith('[open]'):
                    elements = plan[0].split(' ')
                    self.last_opened = [elements[1], elements[2]]
                return None, plan, [last_subgoal]
        """TODO: what if the predicte has been fulfilled but still grabbing the object?"""
        for edge in curr_state_tmp['edges']:
            if edge['relation_type'].startswith('HOLDS') \
                and self.agent_id in [edge['from_id'], edge['to_id']] and last_subgoal.split('_')[0] != 'grab':
                heuristic = heuristic_dict[last_subgoal.split('_')[0]]
                actions, costs = heuristic(self.agent_id, self.char_index, unsatisfied, curr_state_tmp, self.env, last_subgoal)
                plan = [self.get_action_str(action) for action in actions] 
                if len(plan) > 0:
                    elements = plan[0].split(' ') 
                    if need_to_close and (elements[0] == '[walk]' or elements[0] == '[open]' and elements[2] != self.last_opened[1]):
                        if self.last_opened is not None:
                            for edge in curr_state_tmp['edges']:
                                if edge['relation_type'] == 'CLOSE' and \
                                    ('({})'.format(edge['from_id']) == self.last_opened[1] and edge['to_id'] == self.agent_id or \
                                    '({})'.format(edge['to_id']) == self.last_opened[1] and edge['from_id'] == self.agent_id):
                                    plan = ['[close] {} {}'.format(self.last_opened[0], self.last_opened[1])] + plan
                                    break
                    if self.verbose:
                        print(plan[0])
                if len(plan) > 0 and plan[0].startswith('[open]'):
                    elements = plan[0].split(' ')
                    self.last_opened = [elements[1], elements[2]]           
                return None, plan, [last_subgoal]

        self.heuristic_dict = heuristic_dict
        if not curr_root.is_expanded:
            curr_root = self.expand(curr_root, t)

        for explore_step in tqdm(range(self.num_simulation)):
            curr_node = curr_root
            node_path = [curr_node]
            next_node = None

            while curr_node.is_expanded:
                #print('Select', curr_node.id.keys())
                next_node = self.select_child(curr_node)
                if next_node is None:
                    break
                node_path.append(next_node)
                curr_node = next_node

            if next_node is None:
                continue

            leaf_node = self.expand(curr_node, t)

            value = self.rollout(leaf_node, t)
            num_actions = leaf_node.id[1][-2]
            self.backup(value*(self.discount**num_actions), node_path)

        next_root = None
        plan = []
        subgoals = []
        while curr_root.is_expanded:
            actions_taken, children_visit, next_root = self.select_next_root(curr_root)
            curr_root = next_root
            plan += actions_taken
            subgoals.append(next_root.id[0])
        if len(plan) > 0:
            elements = plan[0].split(' ')
            if need_to_close and (elements[0] == '[walk]' or elements[0] == '[open]' and elements[2] != self.last_opened[1]):
                if self.last_opened is not None:
                    for edge in curr_state_tmp['edges']:
                        if edge['relation_type'] == 'CLOSE' and \
                            ('({})'.format(edge['from_id']) == self.last_opened[1] and edge['to_id'] == self.agent_id or \
                            '({})'.format(edge['to_id']) == self.last_opened[1] and edge['from_id'] == self.agent_id):
                            plan = ['[close] {} {}'.format(self.last_opened[0], self.last_opened[1])] + plan
                            break
            if self.verbose:
                print(plan[0])
        if len(plan) > 0 and plan[0].startswith('[open]'):
            elements = plan[0].split(' ')
            self.last_opened = [elements[1], elements[2]]
        return next_root, plan, subgoals


    def rollout(self, leaf_node, t):
        reached_terminal = False

        leaf_node_values = leaf_node.id[1]
        curr_vh_state, curr_state, goal_spec, satisfied, unsatisfied, num_steps, actions_parent = leaf_node_values
        sum_reward = 0
        last_reward = 0
        satisfied = copy.deepcopy(satisfied)
        unsatisfied = copy.deepcopy(unsatisfied)

        # TODO: we should start with goals at random, or with all the goals
        # Probably not needed here since we already computed whern expanding node

        subgoals = self.get_subgoal_space(curr_state, satisfied, unsatisfied, self.opponent_subgoal)
        list_goals = list(range(len(subgoals)))
        random.shuffle(list_goals)
        for rollout_step in range(min(len(list_goals), self.max_rollout_step)):#min(self.max_rollout_step, self.max_episode_length - t)):
            # # subgoals = self.get_subgoal_space(curr_state, satisfied, unsatisfied)
            # print(rollout_step)
            # print(len(list_goals))
            # print(list_goals[rollout_step])
            # print(subgoals)
            # print(subgoals[list_goals[rollout_step]])
            goal_selected = subgoals[list_goals[rollout_step]][0]
            heuristic = self.heuristic_dict[goal_selected.split('_')[0]]
            actions, costs = heuristic(self.agent_id, self.char_index, unsatisfied, curr_state, self.env, goal_selected)
            
            # print(actions)

            if actions is None:
                delta_reward = 0
            elif len(actions) == 0:
                delta_reward = 0
            else:
                num_steps += len(actions)
                cost = sum(costs)
                # print(cost)
                for action_id, action in enumerate(actions):
                    # Check if action can be performed
                    # if action_performed:
                    action_str = self.get_action_str(action)
                    try:
                        next_vh_state = self.env.transition(curr_vh_state, {0: action_str})
                    except:
                        ipdb.set_trace()
                    next_state = next_vh_state.to_dict()
                    curr_vh_state, curr_state = next_vh_state, next_state

                curr_reward = self.check_progress(next_state, goal_spec) # self.env.reward(0, next_state)
                delta_reward = curr_reward - last_reward - cost
                delta_reward = delta_reward #* self.discount**(len(actions))
                # print(curr_rewward, last_reward)
                last_reward = curr_reward
            sum_reward += delta_reward
            # curr_state = next_state
    
        # print(sum_reward, reached_terminal)
        return sum_reward


    def calculate_score(self, curr_node, child):
        parent_visit_count = curr_node.num_visited
        self_visit_count = child.num_visited
        subgoal_prior = child.subgoal_prior

        if self_visit_count == 0:
            u_score = 1e6 #np.inf
            q_score = 0
        else:
            exploration_rate = np.log((1 + parent_visit_count + self.c_base) /
                                      self.c_base) + self.c_init
            u_score = exploration_rate * subgoal_prior * np.sqrt(
                parent_visit_count) / float(1 + self_visit_count)
            q_score = child.sum_value / self_visit_count

        score = q_score + u_score
        return score


    def select_child(self, curr_node):
        scores = [
            self.calculate_score(curr_node, child)
            for child in curr_node.children
        ]
        if len(scores) == 0: return None
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = random.choice(maxIndex)
        selected_child = curr_node.children[selected_child_index]
        return selected_child


    def get_subgoal_prior(self, subgoal_space):
        subgoal_space_size = len(subgoal_space)
        subgoal_prior = {
            subgoal: 1.0 / subgoal_space_size
            for subgoal in subgoal_space
        }
        return subgoal_prior


    def expand(self, leaf_node, t):
        curr_state = leaf_node.id[1][1]
        if t < self.max_episode_length:
            expanded_leaf_node = self.initialize_children(leaf_node)
            if expanded_leaf_node is not None:
                leaf_node.is_expanded = True
                leaf_node = expanded_leaf_node
        return leaf_node


    def backup(self, value, node_list):
        for node in node_list:
            node.sum_value += value
            node.num_visited += 1
            # if value > 0:
            #     print(value, [node.id.keys() for node in node_list])
            # print(value, [node.id.keys() for node in node_list])


    def select_next_root(self, curr_root):
        children_ids = [child.id[0] for child in curr_root.children]
        children_visit = [child.num_visited for child in curr_root.children]
        children_value = [child.sum_value for child in curr_root.children]

        if self.verbose:
            print('children_ids:', children_ids)
            print('children_visit:', children_visit)
            print('children_value:', children_value)
        # print(list([c.id.keys() for c in curr_root.children]))
        maxIndex = np.argwhere(
            children_visit == np.max(children_visit)).flatten()
        selected_child_index = random.choice(maxIndex)
        actions = curr_root.children[selected_child_index].id[1][-1]
        return actions, children_visit, curr_root.children[selected_child_index]

    def transition_subgoal(self, satisfied, unsatisfied, subgoal):
        """transition on predicate level"""
        elements = subgoal.split('_')
        if elements[0] == 'put':
            predicate_key = 'on_{}_{}'.format(self.env.id2node[int(elements[1])], elements[2])
            predicate_value = 'on_{}_{}'.format(elements[1], elements[2])
            satisfied[predicate_key].append(satisfied)


    def initialize_children(self, node):
        leaf_node_values = node.id[1]
        vh_state, state, goal_spec, satisfied, unsatisfied, steps, actions_parent = leaf_node_values

        # print('init child, satisfied:\n', satisfied)
        # print('init child, unsatisfied:\n', unsatisfied)

        subgoals = self.get_subgoal_space(state, satisfied, unsatisfied, self.opponent_subgoal)
        # subgoals = [sg for sg in subgoals if sg[0] != self.opponent_subgoal] # avoid repeating
        # print('init child, subgoals:\n', subgoals)
        if len(subgoals) == 0:
            return None

        goals_expanded = 0
        for goal_predicate in subgoals:
            goal, predicate, aug_predicate = goal_predicate[0], goal_predicate[1], goal_predicate[2] # subgoal, goal predicate, the new satisfied predicate
            heuristic = self.heuristic_dict[goal.split('_')[0]]
            actions_heuristic, costs = heuristic(self.agent_id, self.char_index, unsatisfied, state, self.env, goal)
            if actions_heuristic is None:
                continue
            cost = sum(costs)
            # print(goal_predicate, cost)
            next_vh_state = vh_state
            actions_str = []
            for action in actions_heuristic:
                action_str = self.get_action_str(action)
                actions_str.append(action_str)
                # print([edge for edge in state['edges'] if edge['from_id'] == int(goal.split('_')[1])])
                # print(goal_predicate, action_str)
                # TODO: this could just be computed in the heuristics?
                next_vh_state = self.env.transition(next_vh_state, {0: action_str})
            goals_expanded += 1

            next_satisfied = copy.deepcopy(satisfied)
            next_unsatisfied = copy.deepcopy(unsatisfied)
            if aug_predicate is not None:
                next_satisfied[predicate].append(aug_predicate)
            next_unsatisfied[predicate] -= 1

            # goals_remain = [goal_r for goal_r in goals if goal_r != goal]
            Node(parent=node,
                id=(goal, [next_vh_state, next_vh_state.to_dict(), goal_spec, next_satisfied, next_unsatisfied,
                    len(actions_heuristic), actions_str]),
                 num_visited=0,
                 sum_value=0,
                 subgoal_prior=1.0 / len(subgoals),
                 is_expanded=False)

        if goals_expanded == 0:
            return None
        return node

    def get_action_str(self, action_tuple):
        obj_args = [x for x in list(action_tuple)[1:] if x is not None]
        objects_str = ' '.join(['<{}> ({})'.format(x[0], x[1]) for x in obj_args])
        return '[{}] {}'.format(action_tuple[0], objects_str)

    def get_subgoal_space(self, state, satisfied, unsatisfied, opponent_subgoal=None, verbose=0):
        """
        Get subgoal space
        Args:
            state: current state
            satisfied: satisfied predicates
            unsatisfied: # of unstatisified predicates
        Returns:
            subgoal space
        """
        """TODO: add more subgoal heuristics; currently only have (put x y)"""
        # print('get subgoal space, state:\n', state['nodes'])

        obs = self.env._mask_state(state, self.char_index)
        obsed_objs = [node["id"] for node in obs["nodes"]]

        inhand_objects = []
        for edge in state['edges']:
            if edge['relation_type'].startswith('HOLDS') and \
                edge['from_id'] == self.agent_id:
                inhand_objects.append(edge['to_id'])
        inhand_objects_opponent = []
        for edge in state['edges']:
            if edge['relation_type'].startswith('HOLDS') and \
                edge['from_id'] == 3 - self.agent_id:
                inhand_objects_opponent.append(edge['to_id'])

        # if verbose:
        #     print('inhand_objects:', inhand_objects)
        #     print(state['edges'])

        id2node = {node['id']: node for node in state['nodes']}

        opponent_predicate_1 = None
        opponent_predicate_2 = None
        if opponent_subgoal is not None:
            elements = opponent_subgoal.split('_')
            if elements[0] in ['put', 'putIn']:
                obj1_class = None
                for node in state['nodes']:
                    if node['id'] == int(elements[1]):
                        obj1_class = node['class_name']
                        break
                # if obj1_class is None:
                #     opponent_subgoal = None
                # else:
                opponent_predicate_1 = '{}_{}_{}'.format('on' if elements[0] == 'put' else 'inside', obj1_class, elements[2])
                opponent_predicate_2 = '{}_{}_{}'.format('on' if elements[0] == 'put' else 'inside', elements[1], elements[2])

        subgoal_space, obsed_subgoal_space, overlapped_subgoal_space = [], [], []
        for predicate, count in unsatisfied.items():
            if count > 1 or count > 0 and predicate not in [opponent_predicate_1, opponent_predicate_2]:
                elements = predicate.split('_')
                # print(elements)
                if elements[0] == 'on':
                    subgoal_type = 'put'
                    obj = elements[1]
                    surface = elements[2] # assuming it is a graph node id
                    for node in state['nodes']:
                        if node['class_name'] == obj or str(node['id']) == obj:
                            # print(node)
                            # if verbose:
                            #     print(node)
                            tmp_predicate = 'on_{}_{}'.format(node['id'], surface) 
                            if tmp_predicate not in satisfied[predicate]:
                                tmp_subgoal = '{}_{}_{}'.format(subgoal_type, node['id'], surface)
                                if tmp_subgoal != opponent_subgoal:
                                    subgoal_space.append(['{}_{}_{}'.format(subgoal_type, node['id'], surface), predicate, tmp_predicate])
                                    if node['id'] in obsed_objs:
                                        obsed_subgoal_space.append(['{}_{}_{}'.format(subgoal_type, node['id'], surface), predicate, tmp_predicate])
                                    if node['id'] in inhand_objects:
                                        return [subgoal_space[-1]]
                elif elements[0] == 'inside':
                    subgoal_type = 'putIn'
                    obj = elements[1]
                    surface = elements[2] # assuming it is a graph node id
                    for node in state['nodes']:
                        if node['class_name'] == obj or str(node['id']) == obj:
                            # if verbose:
                            #     print(node)
                            tmp_predicate = 'inside_{}_{}'.format(node['id'], surface) 
                            if tmp_predicate not in satisfied[predicate]:
                                tmp_subgoal = '{}_{}_{}'.format(subgoal_type, node['id'], surface)
                                if tmp_subgoal != opponent_subgoal:
                                    subgoal_space.append(['{}_{}_{}'.format(subgoal_type, node['id'], surface), predicate, tmp_predicate])
                                    if node['id'] in obsed_objs:
                                        obsed_subgoal_space.append(['{}_{}_{}'.format(subgoal_type, node['id'], surface), predicate, tmp_predicate])
                                    if node['id'] in inhand_objects:
                                        return [subgoal_space[-1]]
                elif elements[0] == 'offOn':
                    if id2node[elements[2]]['class_name'] in ['dishwasher', 'kitchentable']:
                        containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']]
                    else:
                        containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] == 'coffetable']
                    for edge in state['edges']:
                        if edge['relation_type'] == 'ON' and edge['to_id'] == int(elements[2]) and id2node[edge['from_id']]['class_name'] == elements[1]:
                            container = random.choice(containers)
                            predicate = '{}_{}_{}'.format('on' if container[1] == 'kitchencounter' else 'inside', edge['from_id'], container[0])
                            goals[predicate] = 1
                elif elements[0] == 'offInside':
                    if id2node[elements[2]]['class_name'] in ['dishwasher', 'kitchentable']:
                        containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] in ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']]
                    else:
                        containers = [[node['id'], node['class_name']] for node in state['nodes'] if node['class_name'] == 'coffetable']
                    for edge in state['edges']:
                        if edge['relation_type'] == 'INSIDE' and edge['to_id'] == int(elements[2]) and id2node[edge['from_id']]['class_name'] == elements[1]:
                            container = random.choice(containers)
                            predicate = '{}_{}_{}'.format('on' if container[1] == 'kitchencounter' else 'inside', edge['from_id'], container[0])
                            goals[predicate] = 1
            elif count > 0 and predicate in [opponent_predicate_1, opponent_predicate_2]: #and len(inhand_objects_opponent) == 0: can added, under testing
                elements = predicate.split('_')
                # print(elements)
                if elements[0] == 'on':
                    subgoal_type = 'put'
                    obj = elements[1]
                    surface = elements[2] # assuming it is a graph node id
                    for node in state['nodes']:
                        if node['class_name'] == obj or str(node['id']) == obj:
                            tmp_predicate = 'on_{}_{}'.format(node['id'], surface) 
                            if tmp_predicate not in satisfied[predicate]:
                                tmp_subgoal = '{}_{}_{}'.format(subgoal_type, node['id'], surface)
                                overlapped_subgoal_space.append(['{}_{}_{}'.format(subgoal_type, node['id'], surface), predicate, tmp_predicate])                        
                elif elements[0] == 'inside':
                    subgoal_type = 'putIn'
                    obj = elements[1]
                    surface = elements[2] # assuming it is a graph node id
                    for node in state['nodes']:
                        if node['class_name'] == obj or str(node['id']) == obj:
                            tmp_predicate = 'inside_{}_{}'.format(node['id'], surface) 
                            if tmp_predicate not in satisfied[predicate]:
                                tmp_subgoal = '{}_{}_{}'.format(subgoal_type, node['id'], surface)
                                overlapped_subgoal_space.append(['{}_{}_{}'.format(subgoal_type, node['id'], surface), predicate, tmp_predicate])
                                    
        if len(obsed_subgoal_space) > 0:
            return obsed_subgoal_space
        if len(subgoal_space) == 0:
            # if self.agent_id == 2 and verbose == 1:
            #     ipdb.set_trace()
            if len(overlapped_subgoal_space) > 0:
                return overlapped_subgoal_space
            for predicate, count in unsatisfied.items():
                if count == 1:
                    elements = predicate.split('_')
                    # print(elements)
                    if elements[0] == 'turnOn':
                        subgoal_type = 'turnOn'
                        obj = elements[1]
                        for node in state['nodes']:
                            if node['class_name'] == obj or str(node['id']) == obj:
                                # print(node)
                                # if verbose:
                                #     print(node)
                                tmp_predicate = 'turnOn{}_{}'.format(node['id'], 1) 
                                if tmp_predicate not in satisfied[predicate]:
                                    subgoal_space.append(['{}_{}'.format(subgoal_type, node['id']), predicate, tmp_predicate])
        if len(subgoal_space) == 0:
            for predicate, count in unsatisfied.items():
                if count == 1:
                    elements = predicate.split('_')
                    # print(elements)
                    if elements[0] == 'holds' and int(elements[2]) == self.agent_id:
                        subgoal_type = 'grab'
                        obj = elements[1]
                        for node in state['nodes']:
                            if node['class_name'] == obj or str(node['id']) == obj:
                                # print(node)
                                # if verbose:
                                #     print(node)
                                tmp_predicate = 'holds_{}_{}'.format(node['id'], 1) 
                                if tmp_predicate not in satisfied[predicate]:
                                    subgoal_space.append(['{}_{}'.format(subgoal_type, node['id']), predicate, tmp_predicate])
        if len(subgoal_space) == 0:
            for predicate, count in unsatisfied.items():
                if count == 1:
                    elements = predicate.split('_')
                    # print(elements)
                    if elements[0] == 'sit' and int(elements[1]) == self.agent_id:
                        subgoal_type = 'sit'
                        obj = elements[2]
                        for node in state['nodes']:
                            if node['class_name'] == obj or str(node['id']) == obj:
                                # print(node)
                                # if verbose:
                                #     print(node)
                                tmp_predicate = 'sit_{}_{}'.format(1, node['id']) 
                                if tmp_predicate not in satisfied[predicate]:
                                    subgoal_space.append(['{}_{}'.format(subgoal_type, node['id']), predicate, tmp_predicate])

        return subgoal_space




        
