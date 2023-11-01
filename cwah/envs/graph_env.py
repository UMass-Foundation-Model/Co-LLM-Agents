import numpy as np
import json
import copy
import ipdb
import itertools
import os
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{curr_dir}/../../virtualhome/simulation/')

from evolving_graph.utils import load_graph_dict, graph_dict_helper
from evolving_graph.execution import ScriptExecutor, ExecutionInfo
from evolving_graph.scripts import read_script_from_string

from evolving_graph.environment import EnvironmentGraph, EnvironmentState


class VhGraphEnv():

    metadata = {'render.modes': ['human']}
    action_executors = [

    ]
    actions = [
      "Walk",   # Same as Run
      #"Find", 
      "Sit", 
      "StandUp", 
      "Grab", 
      "Open", 
      "Close", 
      "PutBack", 
      "PutIn", 
      "SwitchOn", 
      "SwitchOff", 
      #"Drink", 
      "LookAt", 
      "TurnTo", 
      #"Wipe", 
      #"Run", 
      "PutOn", 
      "PutOff", 
      #"Greet", 
      "Drop",     # Same as Release
      #"Read", 
      "PointAt", 
      "Touch", 
      "Lie", 
      "PutObjBack", 
      "Pour", 
      #"Type", 
      #"Watch", 
      "Push", 
      "Pull", 
      "Move", 
      #"Rinse", 
      #"Wash", 
      #"Scrub", 
      #"Squeeze", 
      "PlugIn", 
      "PlugOut", 
      "Cut", 
      #"Eat", 
      "Sleep",
      "SendMessage", #sjm add it
      "WakeUp", 
      #"Release"
    ]
    map_properties_to_pred = {
        'ON': ('on', True),
        'OPEN': ('open', True),
        'OFF': ('on', False),
        'CLOSED': ('open', False)
    }
    map_edges_to_pred = {
        'INSIDE': 'inside',
        'CLOSE': 'close',
        'ON': 'ontop',
        'FACING': 'facing'
    }
    house_obj = [
            'floor',
            'wall',
            'ceiling'
    ]

    def __init__(self, n_chars=1, max_nodes=200):
        self.graph_helper = graph_dict_helper()
        self.n_chars = n_chars
        self.name_equivalence = None
        
        self.state = None
        self.observable_state_n = [None for i in range(self.n_chars)]
        self.character_n = [None for i in range(self.n_chars)]
        self.tasks_n = [None for i in range(self.n_chars)]
        self.prev_progress_n = [None for i in range(self.n_chars)]
        self.rooms = None
        self.rooms_ids = None
        self.observable_object_ids_n = [None for i in range(self.n_chars)]
        self.pomdp = False
        self.executor_n = [ScriptExecutor(EnvironmentGraph(self.state), self.name_equivalence, i) for i in range(self.n_chars)]
    

        

    def to_pomdp(self):
        self.pomdp = True
        for i in range(self.n_chars):
            if self.observable_object_ids_n[i] is None:
                self.observable_state_n[i] = self._mask_state(self.state, i)
                self.observable_object_ids_n[i] = [node["id"] for node in self.observable_state_n[i]["nodes"]]

    def to_fomdp(self):
        self.pomdp = False
        self.observable_object_ids = [None for i in range(self.n_chars)]


    def _remove_house_obj(self, state):
        delete_ids = [x['id'] for x in state['nodes'] if x['class_name'].lower() in self.house_obj]
        state['nodes'] = [x for x in state['nodes'] if x['id'] not in delete_ids]
        state['edges'] = [x for x in state['edges'] if x['from_id'] not in delete_ids and x['to_id'] not in delete_ids]
        return state

    def get_observations(self, graph_env=None, char_index=0):
        if graph_env is None:
            state = self.vh_state.to_dict()
        else:
            state = graph_env

        observable_state = self._mask_state(state, char_index) if self.pomdp else state
        return observable_state

    def step(self, scripts):
        obs_n = []
        info_n = {'n':[]}
        reward_n = []

        if self.pomdp:
            for i in range(self.n_chars):
                if i not in scripts:
                    continue
                assert self._is_action_valid(scripts.get(i), i)

        # State transition: Sequentially performing actions
        # TODO: Detect action conflicts
        # convert action to a single action script
        objs_in_use = []
        for i in range(self.n_chars):
            if i not in scripts:
                continue
            script = read_script_from_string(scripts.get(i, ""))

            is_executable, msg = self._is_action_executable(script, i, objs_in_use)
            if (is_executable):
                objs_in_use += script.obtain_objects()
                succeed, self.vh_state = self.executor_n[i].execute_one_step(script, self.vh_state)
                info_n['n'].append({
                    "succeed": succeed, 
                    "error_message": {i: self.executor_n[i].info.get_error_string() for i in range(self.n_chars)}
                })
            else:
                info_n['n'].append({
                    "succeed": False,
                    "error_message": {i: msg}
                })

        state = self.vh_state.to_dict()
        self.state = state
        
        for i in range(self.n_chars):
            observable_state = self._mask_state(state, i) if self.pomdp else state
            self.observable_state_n[i] = observable_state
            self.observable_object_ids_n[i] = [node["id"] for node in observable_state["nodes"]]
            obs_n.append(observable_state)
            # Reward Calculation

            # progress = self.tasks_n[i].measure_progress(self.observable_state_n[i], i)
            #progress_per_task = [task.measure_progress(self.observable_state_n[i], i) for task in self.tasks_n[i]]
            #progress = sum(progress_per_task) / float(len(progress_per_task))
            progress = 0
            reward_n.append(progress - self.prev_progress_n[i])
            self.prev_progress_n[i] = progress

            # if abs(progress - 1.0) < 1e-6:
            #     info_n['n'][i].update({'terminate': True})
            # else:
            #     info_n['n'][i].update({'terminate': False})


        # Information
    
        return reward_n, obs_n, info_n

    def reward(self, agent_id, state):
        #progress_per_task = [task.measure_progress(self._mask_state(state, agent_id), agent_id) for task in self.tasks_n[agent_id]]
        return 0
        #return sum(progress_per_task) / float(len(progress_per_task))

    def transition(self, vh_state, scripts, do_assert=False):
        # print(scripts, self.observable_object_ids_n[0])
        if do_assert:
            if self.pomdp:
                for i in range(self.n_chars):
                    observable_nodes = self._mask_state(vh_state.to_dict(), i)['nodes']
                    observable_object_ids = [node['id'] for node in observable_nodes]
                    assert self._is_action_valid_sim(scripts.get(i), observable_object_ids)
                
        for i in range(self.n_chars):
            script = read_script_from_string(scripts.get(i, ""))
            succeed, next_vh_state = self.executor_n[i].execute_one_step(script, vh_state)

        # state = next_vh_state.to_dict() 
        return next_vh_state

    def get_vh_state(self, state, name_equivalence=None, instance_selection=True):
        if name_equivalence is None:
            name_equivalence = self.name_equivalence

        return EnvironmentState(EnvironmentGraph(state), self.name_equivalence, instance_selection=True)


    def reset_graph(self, state_graph):

        state = self._remove_house_obj(state_graph)
        
        for i in range(self.n_chars):
            self.executor = ScriptExecutor(EnvironmentGraph(state), self.name_equivalence, i)

        self.character_n = [None for i in range(self.n_chars)]
        chars = [node for node in state["nodes"] if node["category"] == "Characters"]
        chars.sort(key=lambda node: node['id']) 
        assert len(chars) == self.n_chars

        self.character_n = chars

        self.rooms = []
        for node in state["nodes"]:
            if node["category"] == "Rooms":
                self.rooms.append(node)
        self.rooms_ids = [n["id"] for n in self.rooms]
        self.state = state
        self.vh_state = self.get_vh_state(state)

        for i in range(self.n_chars):
            observable_state_n = [self._mask_state(state, i) if self.pomdp else state for i in range(self.n_chars)]
            self.observable_state_n = observable_state_n
            self.observable_object_ids_n = [[node['id'] for node in obs_state['nodes']] for obs_state in observable_state_n]



        return observable_state_n


    def fill_missing_states(self, state):
        for node in state['nodes']:
            object_name = node['class_name']
            states_graph_old = node['states']
            bin_vars = self.graph_helper.get_object_binary_variables(object_name)
            bin_vars_missing = [x for x in bin_vars if x.positive not in states_graph_old and x.negative not in states_graph_old]
            states_graph = states_graph_old + [x.default for x in bin_vars_missing]
            # fill out the rest of info regarding the states
            node['states'] = states_graph



    # TODO: Now the random function doesn't align with the manually set seed
    # task_goals_n is a list of list that represents the goals of every agent
    def reset(self, state, task_goals_n):
        ############ State ############

        state = self._remove_house_obj(state)

        # Fill out the missing states
        self.fill_missing_states(state)
        
        for i in range(self.n_chars):
            self.executor = ScriptExecutor(EnvironmentGraph(state), self.name_equivalence, i)

        self.character_n = [None for i in range(self.n_chars)]
        chars = [node for node in state["nodes"] if node["category"] == "Characters"]
        chars.sort(key=lambda node: node['id']) 

        self.character_n = chars

        self.rooms = []
        for node in state["nodes"]:
            if node["category"] == "Rooms":
                self.rooms.append(node)
        self.rooms_ids = [n["id"] for n in self.rooms]
        self.state = state
        self.vh_state = self.get_vh_state(state)

        ############ Reward ############
        observable_state_n = [self._mask_state(state, i) if self.pomdp else state for i in range(self.n_chars)]
        self.observable_state_n = observable_state_n
        self.observable_object_ids_n = [[node['id'] for node in obs_state['nodes']] for obs_state in observable_state_n]




        return observable_state_n

    def is_terminal(self, char_id, state):
        # masked_state = self._mask_state(state, char_id)
        # print('is_terminal:', [e for e in state['edges'] if 2038 in e.values()])
        # print('is_terminal:', [e for e in masked_state['edges'] if 2038 in e.values()])
        # print('is_terminal:', [(n['class_name'], n['id']) for n in masked_state['nodes']])
        #return self.tasks_n[char_id].measure_progress(self._mask_state(state, char_id), char_id)
        # progress_per_task = [task.measure_progress(self._mask_state(state, char_id), char_id) for task in self.tasks_n[char_id]]
        # return all(progress_per_task)
        return False
        return abs(self.reward(char_id, state) - 1) < 1e-6


    def render(self, mode='human', close=False):
        return

    def _is_action_valid(self, string: str, char_index):

        script = read_script_from_string(string)

        valid = True
        for object_and_id in script.obtain_objects():
            id = object_and_id[1]
            if id not in self.observable_object_ids_n[char_index]:
                valid = False
                break

        return valid

    def _is_action_executable(self, script, char_index, objs_in_use):
        # if there's agent already interacting with the object in this step
        for obj in script.obtain_objects():
            if obj in objs_in_use:
                return False, "object <{}> ({}) is interacted by other agent".format(obj[0], obj[1])
        # # if object is held by others
        #     for i in range(self.n_chars):
        #         if i != char_index:
        #             node = Node(obj[1])
        #             if self.vh_state.evaluate(ExistsRelation(CharacterNode(char_index), Relation.HOLDS_RH, NodeInstance(node))) or \
        #                self.vh_state.evaluate(ExistsRelation(CharacterNode(char_index), Relation.HOLDS_LH, NodeInstance(node))):
        #                 return False, "object <{}> ({}) is held by other agent".format(obj[0], obj[1])
        return True, None


    def _is_action_valid_sim(self, string: str, observable_object_ids):

        script = read_script_from_string(string)

        valid = True
        for object_and_id in script.obtain_objects():
            id = object_and_id[1]
            if id not in observable_object_ids:
                valid = False
                break

        return valid
    
    def get_action_space(self, vh_state=None, char_index=0, action=None, obj1=None, obj2=None, structured_actions=False):
        # TODO: this could probably just go into virtualhome

        if vh_state is None:
            vh_state = self.vh_state
            nodes = self.observable_state_n[char_index]['nodes']
        else:
            nodes = self._mask_state(vh_state.to_dict(), char_index)['nodes']
        node_ids = [x['id'] for x in nodes]

        action_executors = self.executor_n[char_index]._action_executors
        
        if obj1 is not None and obj1['id'] not in node_ids: return []
    

        action_list = []
        action_candidates = self.actions if action is None else [action]
        action_list_sep = []
        

        for action in action_candidates:
            curr_action = Action[action.upper()]
            num_params = curr_action.value[1]
            objects = [[] for _ in range(num_params)]
            for param in range(num_params):
                properties_params = curr_action.value[2][param]
                if param == 0:
                    node_candidates = nodes if obj1 not in nodes else [obj1]
                elif param == 1:
                    node_candidates = nodes if obj2 not in nodes else [obj2]
                else:
                    node_candidates = nodes
                # if param == 0:
                #     node_candidates = nodes if obj1 is None else [obj1]
                # elif param == 1:
                #     node_candidates = nodes if obj2 is None else [obj2]
                # else:
                #     node_candidates = nodes
                
                # remove character from candidates
                node_candidates = [x for x in node_candidates if x['class_name'] != 'character']

                # if obj1 is not None and obj1['id'] == 2038:
                #     print('node candidates:', [node['id'] for node in node_candidates])

                for node in node_candidates:
                    if (len(properties_params) == 0 or 
                        len(set(node['properties']).intersection(properties_params)) > 0):
                            objects[param].append(node)
    
            if any([len(x) == 0 for x in objects]):
                continue
            prod = list(itertools.product(*objects))
            for obj_candidates in prod:
                obj_cand_list = list(obj_candidates)
                string_instr = self.obtain_formatted_action(action, obj_cand_list)
                action_list_tuple = [action] + obj_cand_list 
                if action in ['Walk', 'Find', 'Run']:
                    succeed = True
                else:
                    script = read_script_from_string(string_instr)
                    # This fails, it is modifyng the graph
                    succeed = self.executor_n[char_index].check_one_step(script, vh_state)
                    self.executor_n[char_index].info = ExecutionInfo()
                if succeed:
                    action_list.append(string_instr.lower())
                    action_list_sep.append(action_list_tuple)

        if structured_actions:
            return action_list_sep
        else: 
            return action_list

    def obtain_formatted_action(self, action, obj_cand_list, debug=False):
        if len(obj_cand_list) == 0:
            return '[{}]'.format(action)
        if debug:
            import pdb
            pdb.set_trace()
        obj_list = ' '.join(['<{}> ({})'.format(node_obj['class_name'], node_obj['id']) for node_obj in obj_cand_list])
        string_instr = '[{}] {}'.format(action, obj_list)
        return string_instr

    def _mask_state(self, state, char_index):
        # Assumption: inside is not transitive. For every object, only the closest inside relation is recorded
        character = self.character_n[char_index]
        # find character
        character_id = character["id"]
        id2node = {node['id']: node for node in state['nodes']}
        inside_of, is_inside, edge_from = {}, {}, {}


        grabbed_ids = []
        for edge in state['edges']:
                
            if edge['relation_type'] == 'INSIDE':
                
                if edge['to_id'] not in is_inside.keys():
                    is_inside[edge['to_id']] = []
                
                is_inside[edge['to_id']].append(edge['from_id'])
                inside_of[edge['from_id']] = edge['to_id']


            elif 'HOLDS' in edge['relation_type']:
                if edge['from_id'] == character['id']:
                    grabbed_ids.append(edge['to_id'])


        character_inside_ids = inside_of[character_id]
        room_id =  character_inside_ids


        object_in_room_ids = is_inside[room_id]

        # Some object are not directly in room, but we want to add them
        curr_objects = list(object_in_room_ids)
        while len(curr_objects) > 0:
            objects_inside = []
            for curr_obj_id in curr_objects:
                new_inside = is_inside[curr_obj_id] if curr_obj_id in is_inside.keys() else []
                objects_inside += new_inside
            
            object_in_room_ids += list(objects_inside)
            curr_objects = list(objects_inside)
        
        # Only objects that are inside the room and not inside something closed
        # TODO: this can be probably speed up if we can ensure that all objects are either closed or open
        object_hidden = lambda ido: inside_of[ido] not in self.rooms_ids and 'OPEN' not in id2node[inside_of[ido]]['states']
        observable_object_ids = [object_id for object_id in object_in_room_ids if not object_hidden(object_id)] + self.rooms_ids
        observable_object_ids += grabbed_ids
        

        partilly_observable_state = {
            "edges": [edge for edge in state['edges'] if edge['from_id'] in observable_object_ids and edge['to_id'] in observable_object_ids],
            "nodes": [id2node[id_node] for id_node in observable_object_ids]
        }

        return partilly_observable_state
        
    def _find_node_by_id(self, state, id):
        for node in state["nodes"]:
            if node["id"] == id:
                return node
        return None

    def _filter_edge(self, state, filter):

        target = []
        for edge in state["edges"]:
            if filter(edge):
                target.append(edge)
        
        return target if len(target) > 0 else None

    def _filter_node(self, state, filter):
        
        target = []
        for node in state["nodes"]:
            if filter(node):
                target.append(node)
        
        return target if len(target) > 0 else None

    def _find_targets(self, state, from_id, relation, to_id):

        assert sum([from_id == None, relation == None, to_id == None]) <= 1

        target = []
        if from_id is None:
            for e in state["edges"]:
                if e["relation_type"] == relation and e["to_id"] == to_id:
                    target.append(e["from_id"])
            
        elif to_id is None:
            for e in state["edges"]:
                if e["relation_type"] == relation and e["from_id"] == from_id:
                    target.append(e["to_id"])

        return target if len(target) > 0 else None

    def __str__(self):

        s = ""
        for i in range(self.n_chars):
            s += "Character {}".format(self.character_n[i]["id"]) + "\n"
            s += "Task goal: ({})".format(self.task.goal_n[i]) + "\n"
        
        return s


def _test1():

    env = VhGraphEnv()
    task_goals = '(and (ontop phone[247] kitchen_counter_[230]) (inside character[65] dining_room[201]))'
    state_path = '/scratch/gobi1/andrewliao/programs_processed_precond_nograb_morepreconds/init_and_final_graphs/TrimmedTestScene1_graph/results_intentions_march-13-18/file1003_2.json'
    s = env.reset(state_path, task_goals)

    env.to_pomdp()
    r, s, info = env.step("[walk] <dining_room> (201)")
    r, s, info = env.step("[walk] <phone> (247)")
    r, s, info = env.step("[grab] <phone> (247)")
    print(r, info)

    
if __name__ == '__main__':
    import ipdb
    _test1()
