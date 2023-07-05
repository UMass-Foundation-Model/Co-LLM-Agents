import numpy as np
from pathlib import Path
import cv2
import networkx as nx
from PIL import ImageFont, ImageDraw, Image
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import time
import logging
import atexit
from sys import platform
import subprocess
import os
import glob

import pdb
import random
import torch
import torchvision
import vh_graph
from vh_graph.envs import belief, vh_env
from simulation.unity_simulator import comm_unity as comm_unity
import utils_viz

from agents import MCTS_agent, PG_agent
from gym import spaces, envs
import ipdb
from profilehooks import profile

import utils_rl_agent
logger = logging.getLogger("mlagents_envs")

def check_progress(state, goal_spec):
    """TODO: add more predicate checkers; currently only ON"""
    unsatisfied = {}
    satisfied = {}
    id2node = {node['id']: node for node in state['nodes']}
    for key, value in goal_spec.items():
        elements = key.split('_')
        unsatisfied[key] = value if elements[0] not in ['offOn', 'offInside'] else 0
        satisfied[key] = [None] * 2
        satisfied[key]
        satisfied[key] = []
        for edge in state['edges']:
            if elements[0] in ['on', 'inside']:
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
            elif elements[0] == 'offOn':
                if edge['relation_type'].lower() == 'on' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[1] == 'offInisde':
                if edge['relation_type'].lower() == 'inside' and edge['to_id'] == int(elements[2]) and (id2node[edge['from_id']]['class_name'] == elements[1] or str(edge['from_id']) == elements[1]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['from_id'], elements[2])
                    unsatisfied[key] += 1
            elif elements[0] == 'holds':
                if edge['relation_type'].lower().startswith('holds') and id2node[edge['to_id']]['class_name'] == elements[1] and edge['from_id'] == int(elements[2]):
                    predicate = '{}_{}_{}'.format(elements[0], edge['to_id'], elements[2])
                    satisfied[key].append(predicate)
                    unsatisfied[key] -= 1
        if elements == 'sit':
            if 'SITTING' in id2node[1]['states']:
            # elif elements[0] == 'sit':
            #     if edge['relation_type'].lower().startswith('on') and edge['to_id'] == int(elements[2]) and edge['from_id'] == int(elements[1]):
                predicate = '{}_{}_{}'.format(elements[0], 1, elements[2])
                satisfied[key].append(predicate)
                unsatisfied[key] -= 1
    return satisfied, unsatisfied

class UnityEnvWrapper:
    def __init__(self, 
                 env_id,
                 env_copy_id,
                 init_graph=None,
                 base_port=8080, 
                 num_agents=1,
                 recording=False,
                 output_folder=None,
                 file_name_prefix=None,
                 simulator_args={}):

        self.proc = None
        self.timeout_wait = 60

        self.output_folder = output_folder
        self.file_name_prefix = file_name_prefix

        if base_port is not None:
            self.port_number = base_port + env_copy_id 
            self.comm = comm_unity.UnityCommunication(port=str(self.port_number), **simulator_args)
        else:
            self.comm = comm_unity.UnityCommunication()

        self.num_agents = num_agents
        self.graph = None
        self.recording = recording
        self.follow = False
        self.num_camera_per_agent = 6
        self.CAMERA_NUM = 1 # 0 TOP, 1 FRONT, 2 LEFT..
        

        self.comm.reset(env_id)
        if init_graph is not None:
            self.comm.expand_scene(init_graph)

        # Assumption, over initializing the env wrapper, we only use one enviroment id
        # TODO: make sure this is true
        self.offset_cameras = self.comm.camera_count()[1]
        characters = ['Chars/Female1', 'Chars/Male1']
        rooms = random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        for i in range(self.num_agents):
            self.comm.add_character(characters[i])#, initial_room=rooms[i])#, position=[-1, 0, -7])

        graph = self.get_graph()
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}
        #comm.render_script(['<char0> [walk] <kitchentable> (225)'], camera_mode=False, gen_vid=False)
        #comm.render_script(['<char1> [walk] <bathroom> (11)'], camera_mode=False, gen_vid=False)  
        if self.follow:
            if self.recording:
                comm.render_script(['<char0> [walk] <kitchentable> (225)'], 
                                   recording=self.recording, 
                                   gen_vid=False, 
                                   camera_mode='FIRST_PERSON',
                                   output_folder=output_folder,
                                   file_name_prefix=file_name_prefix,
                                   image_synthesis=['normal', 'seg_inst', 'seg_class'],
                                   time_step=5.
                                   )
            else:
                comm.render_script(['<char0> [walk] <kitchentable> (225)'], camera_mode=False, gen_vid=False, time_scale=10)

        self.get_graph()
        #self.test_prep()

    def close(self):
        self.comm.close()

    def reset(self, env_id, init_graph=None):
        self.comm.reset(env_id)
        if init_graph is not None:
            self.comm.expand_scene(init_graph)
        self.offset_cameras = self.comm.camera_count()[1]
        characters = ['Chars/Female1', 'Chars/Male1']
        rooms = random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        for i in range(self.num_agents):
            self.comm.add_character(characters[i])#, initial_room=rooms[i])#, position=[-1, 0, -7])

        graph = self.get_graph()
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}

    def fast_reset(self, env_id, init_graph=None):
        self.comm.fast_reset(env_id)
        if init_graph is not None:
            self.comm.expand_scene(init_graph)
        self.offset_cameras = self.comm.camera_count()[1]
        characters = ['Chars/Female1', 'Chars/Male1']
        rooms = random.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2)
        for i in range(self.num_agents):
            self.comm.add_character(characters[i])#, initial_room=rooms[i])#, position=[-1, 0, -7])

        graph = self.get_graph()
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}
   
    def returncode_to_signal_name(returncode: int):
        """
        Try to convert return codes into their corresponding signal name.
        E.g. returncode_to_signal_name(-2) -> "SIGINT"
        """
        try:
            # A negative value -N indicates that the child was terminated by signal N (POSIX only).
            s = signal.Signals(-returncode)  # pylint: disable=no-member
            return s.name
        except Exception:
            # Should generally be a ValueError, but catch everything just in case.
            return None





    def get_graph(self):
        if True:  # self.graph is None:
            _, self.graph = self.comm.environment_graph()


        return self.graph

    # TODO: put in some utils
    def world2im(self, camera_data, wcoords):
        wcoords = wcoords.transpose()
        if len(wcoords.shape) < 2:
            return None
        proj = np.array(camera_data['projection_matrix']).reshape((4,4)).transpose()
        w2cam = np.array(camera_data['world_to_camera_matrix']).reshape((4,4)).transpose()
        cw = np.concatenate([wcoords, np.ones((1, wcoords.shape[1]))], 0) # 4 x N
        pixelcoords = np.matmul(proj, np.matmul(w2cam, cw)) # 4 x N
        pixelcoords = pixelcoords/pixelcoords[-1, :]
        pixelcoords = (pixelcoords + 1)/2.
        pixelcoords[1,:] = 1. - pixelcoords[1, :]
        return pixelcoords[:2, :]

    def get_visible_objects(self):
        obj_pos = None
        graph = self.graph

        # Get the objects visible by the character (currently buggy...)
        camera_ids = [[self.offset_cameras+i*self.num_camera_per_agent+self.CAMERA_NUM for i in range(self.num_agents)][1]]
        object_ids = [int(idi) for idi in self.comm.get_visible_objects(camera_ids)[1].keys()]
        _, cam_data = self.comm.camera_data(camera_ids)
        object_position = np.array(
                [node['bounding_box']['center'] for node in graph['nodes'] if node['id'] in object_ids])
        obj_pos = self.world2im(cam_data[0], object_position)


        return object_ids, obj_pos

    def get_observations(self, mode='normal', image_width=128, image_height=128):
        camera_ids = [[self.offset_cameras+i*self.num_camera_per_agent+self.CAMERA_NUM for i in range(self.num_agents)][1]]
        s, images = self.comm.camera_image(camera_ids, mode=mode, image_width=image_width, image_height=image_height)
        #images = [image[:,:,::-1] for image in images]
        return images

    def test_prep(self):
        node_id_new = 2007
        s, graph = self.comm.environment_graph()
        table_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
        container_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['fridge', 'freezer']][0]
        drawer_id = [node['id'] for node in graph['nodes'] if node['class_name'] in ['kitchencabinets']][0]


        id2node = {node['id']: node for node in graph['nodes']}

        # plates = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == table_id and id2node[edge['from_id']]['class_name'] == 'plate']
        # graph['edges'] = [edge for edge in graph['edges'] if edge['from_id'] not in plates and edge['to_id'] not in plates]
        # edge_plates = [{'from_id': plate_id, 'to_id': drawer_id, 'relation_type': 'INSIDE'} for plate_id in plates] 
        # graph['edges'] += edge_plates
        #self.comm.render_script(['<char0> [walk] <livingroom> (319)'], image_synthesis=[]).set_trace()


        new_node = {'id': node_id_new, 'class_name': 'glass', 'states': [], 'properties': ['GRABBABLE']}
        new_edge = {'from_id': node_id_new, 'relation_type': 'INSIDE', 'to_id': container_id}
        graph['nodes'].append(new_node)
        graph['edges'].append(new_edge)
        success = self.comm.expand_scene(graph)
        if self.env_copy_id == 0:
          print(success)

    def agent_ids(self):
        return sorted([x['id'] for x in self.graph['nodes'] if x['class_name'] == 'character'])

    def set_record(self, 
                   output_folder,
                   file_name_prefix):
        self.output_folder = output_folder
        self.file_name_prefix = file_name_prefix

    def execute(self, actions): # dictionary from agent to action
        # Get object to interact with

        # This solution only works for 2 agents, we can scale it for more agents later

        agent_do = list(actions.keys())
        if self.follow:
            actions[0] = '[walk] <character> (438)'
        if len(actions.keys()) > 1:
            if sum(['walk' in x for x in actions.values()]) == 0:
                #continue
                objects_interaction = [x.split('(')[1].split(')')[0] for x in actions.values()]
                if len(set(objects_interaction)) == 1:
                    agent_do = [1] # [random.choice([0,1])]

        script_list = ['']
        for agent_id in agent_do:
            script = actions[agent_id]

            current_script = ['<char{}> {}'.format(agent_id, script)]
            

            script_list = [x+ '|' +y if len(x) > 0 else y for x,y in zip (script_list, current_script)]

        #if self.follow:
        script_list = [x.replace('[walk]', '[walktowards]') for x in script_list]
        # script_all = script_list
        self.graph = None
        if self.recording:
            success, message = self.comm.render_script(script_list,
                                                       recording=True, 
                                                       gen_vid=False, 
                                                       camera_mode='PERSON_TOP',
                                                       output_folder=self.output_folder,
                                                       file_name_prefix=self.file_name_prefix,
                                                       image_synthesis=['normal', 'seg_inst', 'seg_class'])
        else:
            # try:
            success, message = self.comm.render_script(script_list, recording=False, gen_vid=False, processing_time_limit=20, time_scale=10.)
            # except:
            #     success = False
            #     message = {}
        # if not success:
        #     print('action failed:', message)
        #     # ipdb.set_trace()
        result = {}
        for agent_id in agent_do:
            result[agent_id] = (success, message) 
        result['actions'] = script_list
        return result

    def is_terminal(self, goal_spec):
        _, unsatisfied = check_progress(self.graph, goal_spec)
        for predicate, count in unsatisfied.items():
            if count > 0:
                return False
        return True




class UnityEnv:
    def __init__(self, 
                 num_agents=2, 
                 seed=0, 
                 env_id=0, 
                 env_copy_id=0,
                 init_graph=None,
                 observation_type='rgb',
                 max_episode_length=100,
                 enable_alice=True,
                 test_mode=False,
                 simulator_type='python',
                 env_task_set=[],
                 task_type='complex',
                 max_num_objects=150,
                 logging=False,
                 logging_graphs=False,
                 recording=False,
                 record_dir=None,
                 base_port=8080,
                 simulator_args={}):

        self.count_test = 0
        self.test_mode = test_mode
        self.observation_type = observation_type
        self.env_id = env_id
        self.simulator_args = simulator_args
        self.base_port = base_port
        self.enable_alice = enable_alice
        self.task_type = task_type
        self.env_name = 'virtualhome'
        self.num_agents = num_agents
        self.env = vh_env.VhGraphEnv(n_chars=self.num_agents)
        self.env_copy_id = env_copy_id
        self.max_episode_length = max_episode_length
        self.simulator_type = simulator_type
        self.init_graph = init_graph
        self.task_goal = None
        self.env_task_set = env_task_set
        self.logging = logging
        self.logging_graphs = logging_graphs
        self.recording = recording
        self.record_dir = record_dir

        self.unity_simulator = None # UnityEnvWrapper(int(env_id), int(env_copy_id), num_agents=self.num_agents)
        self.agent_ids =  [1,2] # self.unity_simulator.agent_ids()
        self.agents = {}

        self.system_agent_id = self.agent_ids[0]
        self.last_actions = [None] * self.num_agents
        self.last_subgoals = [None] * self.num_agents
        self.task_goal, self.goal_spec = {0: {}, 1: {}}, {}

        self.obj2action = {}

        if self.num_agents>1:
            self.my_agent_id = self.agent_ids[1]

        self.add_system_agent()

        self.actions = {}
        self.actions['system_agent'] = []
        self.actions['my_agent'] = []
        self.image_width = 224
        self.image_height = 224
        self.graph_helper = utils_rl_agent.GraphHelper(max_num_objects=max_num_objects,
                                                       simulator_type=simulator_type)


        ## ------------------------------------------------------------------------------------        
        self.observation_type = observation_type # Image, Coords
        self.viewer = None
        self.num_objects = max_num_objects

        num_actions = len(self.graph_helper.action_dict)
        num_object_classes = len(self.graph_helper.object_dict)
        self.action_space = spaces.Tuple((spaces.Discrete(num_actions), spaces.Discrete(self.num_objects)))



        if self.simulator_type == 'unity':

            self.observation_space = spaces.Dict({
                # Image
                'image': spaces.Box(low=0, high=255., shape=(3, self.image_height, self.image_width)),
                # Graph
                #utils_rl_agent.GraphSpace(),

                'class_objects': spaces.Box(low=0, high=self.graph_helper.num_classes, shape=(self.graph_helper.num_objects, )),
                'node_ids': spaces.Box(low=0, high=500,
                                            shape=(self.graph_helper.num_objects,)),

                'states_objects': spaces.Box(low=0, high=1., shape=(self.graph_helper.num_objects, self.graph_helper.num_states)),
                'edge_tuples': spaces.Box(low=0, high=self.graph_helper.num_objects, shape=(self.graph_helper.num_edges, 2)),
                'edge_classes': spaces.Box(low=0, high=self.graph_helper.num_edge_types, shape=(self.graph_helper.num_edges, )),
                'mask_object': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_objects, )),
                'mask_edge': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_edges, )),

                # Target object
                'object_dist': spaces.Box(low=-100, high=100, shape=(2,)),
                'object_coords': spaces.Box(low=0, high=max(self.image_height, self.image_width),
                           shape=(self.num_objects, 3)), # 3D coords of the objects
                # 'mask_position_objects': spaces.Box(low=0, high=1, shape=(self.num_objects, )),
                'target_obj_class': spaces.Box(low=0, high=num_object_classes, shape=(1, 6)),
                'target_loc_class': spaces.Box(low=0, high=num_object_classes, shape=(1, 6)),
                'affordance_matrix': spaces.Box(low=0, high=1, shape=(num_actions, num_object_classes))
            })

        else:
            self.observation_space = spaces.Dict({
                'class_objects': spaces.Box(low=0, high=self.graph_helper.num_classes, shape=(self.graph_helper.num_objects,)),
                'states_objects': spaces.Box(low=0, high=1., shape=(self.graph_helper.num_objects, self.graph_helper.num_states)),
                'edge_tuples': spaces.Box(low=0, high=self.graph_helper.num_objects, shape=(self.graph_helper.num_edges, 2)),
                'edge_classes': spaces.Box(low=0, high=self.graph_helper.num_edge_types, shape=(self.graph_helper.num_edges,)),
                'mask_object': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_objects,)),
                'mask_edge': spaces.Box(low=0, high=1, shape=(self.graph_helper.num_edges,)),
                'affordance_matrix': spaces.Box(low=0, high=1, shape=(num_actions, num_object_classes))
            })


        self.reward_range = (-10, 50.)
        self.metadata = {'render.modes': ['human']}
        self.spec = envs.registration.EnvSpec('virtualhome-v0')

        
        self.history_observations = []
        self.len_hist = 4
        self.num_steps = 0
        self.prev_dist = None
        self.visible_nodes = None
        self.observed_graph = None

        self.micro_id = -1
        self.last_action = ''

        # The observed nodes

        self.info = {'dist': 0, 'reward': 0}

    def seed(self, seed):
        pass

    def close(self):
        if self.unity_simulator is not None:
            self.unity_simulator.close()

    def distance_reward(self, graph, target_class="microwave"):
        dist, is_close = self.get_distance(graph, target_class=target_class)


        reward = 0#- 0.02
        #print(self.prev_dist, dist, reward)
        self.prev_dist = dist
        #is_done = is_close
        is_done = False
        if is_close:
            reward += 0.05#1 #5
        info = {'dist': dist, 'done': is_done, 'reward': reward}
        return reward, is_close, info


    def reward(self, visible_ids=None, graph=None):
        '''
        goal format:
        {predicate: number}
        predicate format:
            on_objclass_id
            inside_objclass_id
        '''

        # Low level policy reward


        done = False
        if self.task_type == 'find':
            id2node = {node['id']: node for node in graph['nodes']}
            grabbed_obj = [id2node[edge['to_id']]['class_name'] for edge in graph['edges'] if
                           'HOLDS' in edge['relation_type']]
            print(grabbed_obj)
            reward, is_close, info = self.distance_reward(graph, self.goal_find_spec)
            if visible_ids is not None:

                # Reward if object is seen
                if self.level > 0:
                    if len(set(self.goal_find_spec).intersection([node[0] for node in visible_ids])) > 0:
                        reward += 0.5

                if len(set(self.goal_find_spec).intersection(grabbed_obj)) > 0.:
                    reward += 1#100.
                    done = True
            return reward, done, info

        elif self.task_type == 'open':
            raise NotImplementedError
            reward, is_close, info = self.distance_reward(graph, self.goal_find_spec)
            return reward, done, info

        if self.simulator_type == 'unity':
            satisfied, unsatisfied = check_progress(self.get_graph(), self.goal_spec)

        else:
            satisfied, unsatisfied = check_progress(self.env.state, self.goal_spec)


        # print('reward satisfied:', satisfied)
        # print('reward unsatisfied:', unsatisfied)
        # print('reward goal spec:', self.goal_spec)
        count = 0
        done = True
        for key, value in satisfied.items():
            count += min(len(value), self.goal_spec[key])
            if unsatisfied[key] > 0:
                done = False
        return count, done, {}
    

    def get_distance(self, graph=None, target_id=None, target_class=['microwave'], norm=None):
        is_close = False
        if self.simulator_type == 'unity':
            if graph is None:
                gr = self.unity_simulator.get_graph()
            else:
                gr = graph

            if target_id is None:
                try:
                    char_node = [node['bounding_box']['center'] for node in gr['nodes'] if node['class_name'] == 'character' and node['id'] == self.my_agent_id][0]
                    target_node_id = [node['id'] for node in gr['nodes'] if node['class_name'] in target_class]
                    target_id = target_node_id
                except:
                    pdb.set_trace()
            target_node = [node['bounding_box']['center'] for node in gr['nodes'] if node['id'] in target_id]
            if graph is not None:
                if len([edge for edge in graph['edges'] if
                        edge['from_id'] in target_id and edge['to_id'] == self.my_agent_id]) > 0:
                    is_close = True

            if norm == 'no':
                return np.array(char_node) - np.array(target_node[0]), is_close
            dist = (np.linalg.norm(np.array(char_node) - np.array(target_node[0]), norm))
            #print([node['id'] for node in gr['nodes'] if node['class_name'] == 'microwave'])
            # print(dist, char_node, micro_node)

        else:
            gr = self.env.state
            if target_id is None:
                target_id = [node['id'] for node in gr['nodes'] if node['class_name'] in target_class][0]

            if len([edge for edge in gr['edges'] if edge['from_id'] == target_id and edge['to_id'] == self.my_agent_id]) > 0:
                dist = 0
            else:
                dist = 5.
                is_close = True
        return dist, is_close


    def render_visdom(self):

        utils_viz.plot_graph(self.observed_graph, self.visible_nodes)


    def render(self, mode='human'):
        image_width = 500
        image_height = 500
        obs, img = self.get_observations(mode='normal', image_width=image_width, image_height=image_height, drawing=True)

        fig = plt.figure()
        graph_viz = img[1][0].to_networkx()
        nx.draw(graph_viz, with_labels=True, labels=img[1][1])
        plt.show()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        img_graph = cv2.resize(image_from_plot, (image_width, image_height))
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        im_pil = Image.fromarray(img[0])

        draw = ImageDraw.Draw(im_pil)
        # Choose a font
        font = ImageFont.truetype("Roboto-Regular.ttf", 30)

        # Draw the text
        draw.text((0, 0), "dist: {:.3f}".format(self.info['dist']), font=font)
        draw.text((0, 60), "reward {:.3f}".format(self.info['reward']), font=font)
        draw.text((0, 90), "last action: {}".format(self.last_action), font=font)

        img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        img = np.concatenate([img, img_graph], 1)


        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=500)
            self.viewer.imshow(img)
            return self.viewer.isopen
         

    def reset(self, graph=None, task_goal=None):
        # reset system agent
        # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]

        # pdb.set_trace()
        if self.test_mode:
            env_task = self.env_task_set[self.count_test]
        else:
            env_task = random.choice(self.env_task_set)
        self.task_id = env_task['task_id']
        self.init_graph = env_task['init_graph']
        self.init_rooms = env_task['init_rooms']
        self.task_goal = env_task['task_goal']
        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']
        self.goal_spec = self.task_goal[self.system_agent_id]

        random.seed(self.task_id)
        np.random.seed(self.task_id)

        self.obj2action = {}

        # Select an object ar random from our tasks
        objects_spec = list(self.goal_spec.keys())
        object_goal = random.choice(objects_spec)

        if self.task_type == 'find':
            self.goal_find_spec = [object_goal.split('_')[1]]
            print('Goal: {}'.format(self.goal_find_spec))
        elif self.task_type == "open":
            self.goal_find_spec = ['microwave', 'fridge', 'cabinet', 'kitchencabinets']
        else:
            self.goal_find_spec = []
        self.level = 1#env_task['level']

        self.graph_helper.get_action_affordance_map(self.task_goal, {node['id']: node for node in self.init_graph['nodes']})
        print('env_id:', self.env_id)
        print('task_name:', self.task_name)
        print('goals:', self.task_goal[0])

        if self.simulator_type == 'unity':
            record_dir = self.record_dir

            if self.recording:
                Path(record_dir).mkdir(parents=True, exist_ok=True)
                file_name_prefix = str(self.task_id) + '_' + self.task_name
            else:
                record_dir = 'Output'
                file_name_prefix = None
            if self.unity_simulator is None:
                self.unity_simulator = UnityEnvWrapper(int(self.env_id), int(self.env_copy_id),
                                                       init_graph=self.init_graph,
                                                       num_agents=self.num_agents,
                                                       base_port=self.base_port,
                                                       recording=self.recording,
                                                       output_folder=record_dir + '/',
                                                       file_name_prefix=file_name_prefix,
                                                       simulator_args=self.simulator_args)
                self.unity_simulator.reset(self.env_id, self.init_graph)
            else:
                self.unity_simulator.set_record(output_folder=record_dir + '/', file_name_prefix=file_name_prefix)
            # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
            # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]	        if graph is None:
                if False:
                    self.unity_simulator.comm.fast_reset(self.env_id)
                else:
                    self.unity_simulator.reset(self.env_id, self.init_graph)


            curr_graph_system_agent = self.inside_not_trans(self.unity_simulator.get_graph())
            self.env.reset(curr_graph_system_agent, self.task_goal)
            self.env.to_pomdp()
            self.init_unity_graph = self.get_unity_graph()

        else:
            # room_ids = [node['id'] for node in self.init_graph['nodes'] if node['category'] == 'Rooms']
            # random.choices(room_ids, k=2)
            room_ids = list(self.init_rooms)
            # ipdb.set_trace()
            char_positions = [{'from_id': 1,  'to_id': room_ids[0], 'relation_type': 'INSIDE'},
                              {'from_id': 2,  'to_id': room_ids[1], 'relation_type': 'INSIDE'}]
            dummybbox = {'center': [0,0,0], 'size': [0,0,0]}
            char_nodes = [{'id': 1, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []},
                          {'id': 2, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []}]
            init_graph_chars = {
                'edges': self.init_graph['edges'] + char_positions,
                'nodes': self.init_graph['nodes'] + char_nodes,
            }
            init_graph_chars['edges'] = [edge for edge in init_graph_chars['edges'] if edge['relation_type'] != 'CLOSE']
            graph = self.inside_not_trans(init_graph_chars)
            # print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # print('unity env graph:', [edge for edge in self.init_graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            obs_n = self.env.reset(graph, self.task_goal)
            self.env.to_pomdp()
            curr_graph_system_agent = graph
            self.init_unity_graph = None
            # print('unity env graph:', [edge for edge in self.env.state['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # ipdb.set_trace()

        obs = self.get_observations()[0]

        self.agents[self.system_agent_id].reset(curr_graph_system_agent,
                                                self.task_goal,
                                                seed=self.system_agent_id,

                                                simulator_type=self.simulator_type,
                                                is_alice=True)
        self.prev_dist = self.get_distance()

        self.num_steps = 0
        # pdb.set_trace()
        return obs

    def reset_2agents_python(self):
        env_task = random.choice(self.env_task_set)
        self.init_graph = env_task['init_graph']
        self.task_goal = env_task['task_goal']
        self.task_name = env_task['task_name']
        self.env_id = env_task['env_id']
        self.goal_spec = self.task_goal[self.system_agent_id]
        print('env_id:', self.env_id)
        print('task_name:', self.task_name)
        print('goals:', self.task_goal[0])
        if self.unity_simulator is None:
            self.unity_simulator = UnityEnvWrapper(int(self.env_id), int(self.env_copy_id), init_graph=self.init_graph, num_agents=self.num_agents,
                                                       file_name=self.file_name)
        graph = self.inside_not_trans(self.unity_simulator.get_graph())
        obs_n = self.env.reset(graph, self.task_goal)

        # pdb.set_trace()
        self.agents[self.system_agent_id].reset(graph, self.task_goal, seed=self.system_agent_id)
        self.prev_dist = self.get_distance()[0]
        self.num_steps = 0
        return obs_n

    def reset_MCTS(self, graph=None, task_goal=None, task_id=None):
        # reset system agent
        # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
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
        self.goal_spec = self.task_goal[self.system_agent_id]
        self.level = env_task['level']
        random.seed(self.task_id)
        np.random.seed(self.task_id)

        self.graph_helper.get_action_affordance_map(self.task_goal, {node['id']: node for node in self.init_graph['nodes']})
        print('task_id:', self.task_id)
        print('env_id:', self.env_id)
        print('task_name:', self.task_name)
        print('goals:', self.task_goal[0])

        self.obj2action = {}


        if self.simulator_type == 'unity':
            record_dir = self.record_dir
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            file_name_prefix = str(self.task_id) + '_' + self.task_name

            if self.unity_simulator is None:
                self.unity_simulator = UnityEnvWrapper(int(self.env_id), int(self.env_copy_id),
                                                       init_graph=self.init_graph,
                                                       num_agents=self.num_agents,
                                                       base_port=self.base_port,
                                                       recording=self.recording,
                                                       output_folder=record_dir + '/',
                                                       file_name_prefix=file_name_prefix,
                                                       simulator_args = self.simulator_args)

            else:
                self.unity_simulator.set_record(output_folder=record_dir + '/', file_name_prefix=file_name_prefix)
            # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
            # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]           if graph is None:
            self.unity_simulator.reset(self.env_id, self.init_graph)

            #self.env.reset(self.init_graph, self.task_goal)
            curr_graph_system_agent = self.inside_not_trans(self.unity_simulator.get_graph())
            self.init_unity_graph = self.get_unity_graph()

        else:
            # room_ids = [node['id'] for node in self.init_graph['nodes'] if node['category'] == 'Rooms']
            # random.choices(room_ids, k=2)
            room_ids = list(self.init_rooms)
            # ipdb.set_trace()
            char_positions = [{'from_id': 1,  'to_id': room_ids[0], 'relation_type': 'INSIDE'},
                              {'from_id': 2,  'to_id': room_ids[1], 'relation_type': 'INSIDE'}]
            dummybbox = {'center': [0,0,0], 'size': [0,0,0]}
            char_nodes = [{'id': 1, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []},
                          {'id': 2, 'class_name': 'character', 'bounding_box': dummybbox, 'states': [], 'category': 'Characters', 'properties': []}]
            init_graph_chars = {
                'edges': self.init_graph['edges'] + char_positions,
                'nodes': self.init_graph['nodes'] + char_nodes,
            }
            init_graph_chars['edges'] = [edge for edge in init_graph_chars['edges'] if edge['relation_type'] != 'CLOSE']
            graph = self.inside_not_trans(init_graph_chars)
            # print('unity env graph:', [edge for edge in graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # print('unity env graph:', [edge for edge in self.init_graph['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            obs_n = self.env.reset(graph, self.task_goal)
            self.env.to_pomdp()
            curr_graph_system_agent = graph
            self.init_unity_graph = None
            # print('unity env graph:', [edge for edge in self.env.state['edges'] if edge['from_id'] == 1010 or edge['to_id'] == 1010])
            # ipdb.set_trace()

        self.agents[self.system_agent_id].reset(curr_graph_system_agent, self.task_goal, seed=self.system_agent_id, simulator_type=self.simulator_type, is_alice=True)
        obs = None
        self.num_steps = 0
        # pdb.set_trace()
        return obs

    def reset_alice(self, graph=None, task_goal=None):
        # reset system agent
        # #self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        # #self.history_observations = [torch.zeros(1, 84, 84) for _ in range(self.len_hist)]
        if graph is None:
            self.unity_simulator.comm.fast_reset(self.env_id)
        # #self.unity_simulator.comm.add_character()
        # #self.unity_simulator.comm.render_script(['<char0> [walk] <kitchentable> (225)'], gen_vid=False, recording=True)
        
        if task_goal is not None:
            self.goal_spec = task_goal[self.system_agent_id]
            self.task_goal = task_goal
            self.agents[self.system_agent_id].reset(graph, task_goal, seed=self.system_agent_id)
        self.prev_dist = self.get_distance()[0]
        # obs = self.get_observations()[0]
        obs = None
        self.num_steps = 0
        return obs


    def get_action_command(self, my_agent_action):

        if my_agent_action is None:
            return None

        if self.simulator_type == 'unity':
            current_graph = self.unity_simulator.get_graph()
        else:
            current_graph = self.env.state

        objects1 = self.visible_nodes
        action_id, object_id = my_agent_action

        if type(action_id) != int:
            action_id = action_id.item()

        if type(object_id) != int:
            object_id = object_id.item()

        action = self.graph_helper.action_dict.get_el(action_id)
        (o1, o1_id) = objects1[object_id]
        #action_str = actions[my_agent_action]
        if o1 == 'no_obj':
            o1 = None
    
        converted_action = utils_rl_agent.can_perform_action(action, o1, o1_id, self.my_agent_id, current_graph)

        return converted_action

    def step(self, my_agent_action):
        #actions = ['<char0> [walktowards] <microwave> ({})'.format(self.micro_id), '<char0> [turnleft]', '<char0> [turnright]']
        if self.simulator_type == 'unity':
            action_dict = {}
            # system agent action

            if self.enable_alice:
                graph = self.get_graph()
                # pdb.set_trace()
                if self.num_steps == 0:
                    graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (edge['from_id'] in self.agent_ids or edge['to_id'] in self.agent_ids))]
                self.env.reset(graph , self.task_goal)
                system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal, self.last_actions[0], self.last_subgoals[0])
                self.last_actions[0] = system_agent_action

                if len(system_agent_info['subgoals']) > 0:
                    self.last_subgoals[0] = system_agent_info['subgoals'][0]
                else:
                    self.last_subgoals[0] = None
                # pdb.set_trace()
                if system_agent_action is not None:
                    action_dict[0] = system_agent_action

            # user agent action
            # pdb.set_trace()
            action_str = self.get_action_command(my_agent_action)
            if action_str is not None:
                action_dict[1] = action_str
                elements = action_str.split(' ')
                o1 = int(elements[-1][1:-1])
                self.obj2action[o1] = action_str
            print(action_dict)
            dict_results = self.unity_simulator.execute(action_dict)
            self.num_steps += 1
            obs, info = self.get_observations()

            # pdb.set_trace()
            print(info[-1][2])
            reward, done, info = self.reward(visible_ids=info[-1][2], graph=info[1])
            dict_results['finished'] = done
            reward = torch.Tensor([reward])
            if self.num_steps >= self.max_episode_length:
                done = True
            done = np.array([done])
            graph = self.unity_simulator.get_graph()
            self.env.reset(graph, self.task_goal)
        else:
            action_dict = {}
            if self.enable_alice:
                graph = self.env.state
                if self.num_steps == 0:
                    graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (
                                edge['from_id'] in self.agent_ids or edge['to_id'] in self.agent_ids))]
                self.env.reset(graph, self.task_goal)
                system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal,
                                                                                      self.last_actions[0],
                                                                                      self.last_subgoals[0])
                self.last_actions[0] = system_agent_action
                self.last_subgoals[0] = system_agent_info['subgoals'][0]
                if system_agent_action is not None:
                    action_dict[0] = system_agent_action
            action_str = self.get_action_command(my_agent_action)

            if action_str is not None:
                # if 'walk' not in action_str:
                # print(action_str)
                action_dict[1] = action_str

            _, obs_n, dict_results = self.env.step(action_dict)
            obs, _ = self.get_observations()
            self.num_steps += 1
            reward, done, info = self.reward()
            dict_results['finished'] = done
            reward = reward# - 0.01
            reward = torch.Tensor([reward])
            if self.num_steps >= self.max_episode_length:
                done = True
            done = np.array([done])

        self.last_action = action_str
        return obs, reward, done, dict_results

    def step_2agents_python(self, action_dict):
        _, obs_n, info_n = self.env.step(action_dict)
        self.num_steps += 1
        reward, done, info = self.reward()
        reward = torch.Tensor([reward])
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        return obs_n, reward, done, info_n

    def step_with_system_agent_oracle(self, my_agent_action):
        #actions = ['<char0> [walktowards] <microwave> ({})'.format(self.micro_id), '<char0> [turnleft]', '<char0> [turnright]']
        action_dict = {}
        # system agent action
        graph = self.get_graph()
        # pdb.set_trace()
        if self.num_steps == 0:
            graph['edges'] = [edge for edge in graph['edges'] if not (edge['relation_type'] == 'CLOSE' and (edge['from_id'] in self.agent_ids or edge['to_id'] in self.agent_ids))]
        self.env.reset(graph , self.task_goal)
        system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal, self.last_actions[0], self.last_subgoals[0])
        self.last_actions[0] = system_agent_action
        self.last_subgoals[0] = system_agent_info['subgoals'][0]
        if system_agent_action is not None:
            action_dict[0] = system_agent_action
        system_agent_action, system_agent_info = self.get_system_agent_action(self.task_goal, self.last_actions[0], self.last_subgoals[0])
        self.last_actions[0] = system_agent_action
        self.last_subgoals[0] = system_agent_info['subgoals'][0]
        if system_agent_action is not None:
            action_dict[0] = system_agent_action
        # user agent action
        action_str = my_agent_action
        if action_str is not None:
            print(action_str)
            action_dict[1] = action_str

        dict_results = self.unity_simulator.execute(action_dict)
        self.num_steps += 1
        obs = None
        reward, done, info = self.reward()
        reward = torch.Tensor([reward])
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        # infos = {}
        #if done:
        #    obs = self.reset()
        return obs, reward, done, dict_results

    def step_alice(self):
        if self.simulator_type == 'unity':
            graph = self.get_graph()
            self.env.reset(graph, self.task_goal)
        self.num_steps += 1
        # obs, _ = self.get_observations()
        obs = None
        infos = {}
        reward, done, infos = self.reward()
        infos = {'finished': done}
        reward = torch.Tensor([reward])
        if self.num_steps >= self.max_episode_length:
            done = True
        done = np.array([done])
        
        #if done:
        #    obs = self.reset()
        return obs, reward, done, infos

    def add_system_agent(self):
        ## Alice model
        self.agents[self.system_agent_id] = MCTS_agent(unity_env=self,
                               agent_id=self.system_agent_id,
                               char_index=0,
                               max_episode_length=5,
                               num_simulation=100,
                               max_rollout_steps=3,
                               c_init=0.1,
                               c_base=1000000,
                               num_samples=1,
                               num_processes=1,
                               logging=self.logging,
                               logging_graphs=self.logging_graphs)

    def get_system_agent_action(self, task_goal, last_action, last_subgoal, opponent_subgoal=None):
        # if last_subgoal is not None:
        #     elements = last_subgoal.split('_')
        #     print(elements)
        #     # print(self.agents[self.system_agent_id].belief.edge_belief) #[int(elements[1])]['INSIDE']
        #     ipdb.set_trace()
        self.agents[self.system_agent_id].sample_belief(self.env.get_observations(char_index=0))
        self.agents[self.system_agent_id].sim_env.reset(self.agents[self.system_agent_id].previous_belief_graph, task_goal)
        action, info = self.agents[self.system_agent_id].get_action(task_goal[0], last_action, last_subgoal, opponent_subgoal)
        # if action == '[walk] <cutleryknife> (1010)':
        #     ipdb.set_trace()

        if action is None:
            print("system agent action is None! DONE!")
            # pdb.set_trace()
        # else:
        #     print(action, info['plan'])

        return action, info

    def get_all_agent_id(self):
        return self.agent_ids

    def get_my_agent_id(self):
        if self.num_agents==1:
            error("you haven't set your agent")
        return self.my_agent_id

    def get_graph(self):

        if self.simulator_type == 'unity':
            graph = self.unity_simulator.get_graph()
        else:
            graph = self.env.state
        graph = self.inside_not_trans(graph)
        return graph

    def get_unity_graph(self):
        return self.unity_simulator.get_graph()

    def get_system_agent_observations(self, modality=['rgb_image']):
        observation = self.agents[self.system_agent_id].num_cameras = self.unity_simulator.camera_image(self.system_agent_id, modality)
        return observation

    def get_my_agent_observations(self, modality=['rgb_image']):
        observation = self.agents[self.system_agent_id].num_cameras = self.unity_simulator.camera_image(self.my_agent_id, modality)
        return observation

    def inside_not_trans(self, graph):
        id2node = {node['id']: node for node in graph['nodes']}
        parents = {}
        grabbed_objs = []
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE':

                if edge['from_id'] not in parents:
                    parents[edge['from_id']] = [edge['to_id']]
                else:
                    parents[edge['from_id']] += [edge['to_id']]
            elif edge['relation_type'].startswith('HOLDS'):
                grabbed_objs.append(edge['to_id'])

        edges = []
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
                if len(parents[edge['from_id']]) == 1:
                    edges.append(edge)
                else:
                    if edge['from_id'] > 1000:
                        pdb.set_trace()
            else:
                edges.append(edge)
        graph['edges'] = edges

        # # add missed edges
        # missed_edges = []
        # for obj_id, action in self.obj2action.items():
        #     elements = action.split(' ')
        #     if elements[0] == '[putback]':
        #         surface_id = int(elements[-1][1:-1])
        #         found = False
        #         for edge in edges:
        #             if edge['relation_type'] == 'ON' and edge['from_id'] == obj_id and edge['to_id'] == surface_id:
        #                 found = True
        #                 break
        #         if not found:
        #             missed_edges.append({'from_id': obj_id, 'relation_type': 'ON', 'to_id': surface_id})
        # graph['edges'] += missed_edges


        parent_for_node = {}

        char_close = {1: [], 2:[]}
        for char_id in range(1, 3):
            for edge in graph['edges']:
                if edge['relation_type'] == 'CLOSE':
                    if edge['from_id'] == char_id and edge['to_id'] not in char_close[char_id]:
                        char_close[char_id].append(edge['to_id'])
                    elif edge['to_id'] == char_id and edge['from_id'] not in char_close[char_id]:
                        char_close[char_id].append(edge['from_id'])
        ## Check that each node has at most one parent
        for edge in graph['edges']:
            if edge['relation_type'] == 'INSIDE':
                if edge['from_id'] in parent_for_node and not id2node[edge['from_id']]['class_name'].startswith('closet'):
                    print('{} has > 1 parent'.format(edge['from_id']))
                    pdb.set_trace()
                    raise Exception
                parent_for_node[edge['from_id']] = edge['to_id']
                # add close edge between objects in a container and the character
                if id2node[edge['to_id']]['class_name'] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave', 'dishwasher', 'stove']: 
                    for char_id in range(1, 3):
                        if edge['to_id'] in char_close[char_id] and edge['from_id'] not in char_close[char_id]:
                            graph['edges'].append({
                                    'from_id': edge['from_id'],
                                    'relation_type': 'CLOSE',
                                    'to_id': char_id
                                })
                            graph['edges'].append({
                                    'from_id': char_id,
                                    'relation_type': 'CLOSE',
                                    'to_id': edge['from_id']
                                })

        
        ## Check that all nodes except rooms have one parent
        nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
        nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
        nodes_without_parent = [node for node in nodes_without_parent if node not in grabbed_objs]
        if len(nodes_without_parent) > 0:
            for nd in nodes_without_parent:
                print(id2node[nd])
            pdb.set_trace()
            raise Exception

        return graph

    def get_observations(self, mode='seg_class', image_width=None, image_height=None, drawing=False):
        if self.simulator_type == 'unity':
            if image_height is None:
                image_height = self.image_height
            if image_width is None:
                image_width = self.image_width

            if self.observation_type == 'rgb':
                images = self.unity_simulator.get_observations(mode=mode, image_width=image_width, image_height=image_height)
            else:
                # For this mode we don't need images
                images = [np.zeros((image_width, image_height, 3))]

            current_obs_img = images[0]
            current_obs_img = torchvision.transforms.functional.to_tensor(current_obs_img)[None, :]

            graph = self.unity_simulator.get_graph()

            distance = self.get_distance(norm='no')[0]
            rel_coords = torch.Tensor(list([distance[0], distance[2]]))[None, :]


            if self.observation_type in ['rgb', 'visibleids']:
                visible_objects, position_objects = self.unity_simulator.get_visible_objects()
            elif self.observation_type == 'mcts':
                # Use the mcts function to get the obs
                python_graph = self.env.get_observations(char_index=1)
                visible_objects = [node['id'] for node in python_graph['nodes']]

            elif self.observation_type == 'full':
                visible_objects = [node['id'] for node in self.graph['nodes']]
            else:
                raise NotImplementedError

            id2node = {node['id']: node for node in graph['nodes']}
            visible_objects = [object_id for object_id in visible_objects if
                               self.graph_helper.object_dict.get_id(id2node[object_id]['class_name']) != 0]# and 
                               # id2node[object_id]['class_name'] == 'wineglass']

            if self.level == 0:
                visible_objects = [object for object in visible_objects if id2node[object]['category'] != 'Rooms']
            graph_inputs, graph_viz = self.graph_helper.build_graph(graph, ids=visible_objects,
                                                                    character_id=self.my_agent_id, plot_graph=drawing,
                                                                    level=self.level)
            self.visible_nodes = graph_viz[-1]
            self.observed_graph = graph

            # if self.env_copy_id == 0:
            #     print(self.visible_nodes)

            # mask = torch.Tensor(mask)[None, :]
            for obj_goal in self.goal_find_spec:
                assert(self.graph_helper.object_dict.get_id(obj_goal) > 0)

            current_obs = {'image': current_obs_img}
            current_obs.update(graph_inputs)
            target_obj_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
            target_loc_class = [self.graph_helper.object_dict.get_id('no_obj')] * 6
            pre_id = 0
            for predicate, count in self.goal_spec.items():
                if count == 0:
                    continue
                elements = predicate.split('_')
                obj_class_id = int(self.graph_helper.object_dict.get_id(elements[1]))
                loc_class_id = int(self.graph_helper.object_dict.get_id(id2node[int(elements[2])]['class_name']))
                for tmp_i in range(count):
                    target_obj_class[pre_id] = obj_class_id
                    target_loc_class[pre_id] = loc_class_id
                    pre_id += 1
            current_obs.update(
                {
                    'affordance_matrix': self.graph_helper.obj1_affordance,
                    'object_dist': rel_coords,
                    'target_obj_class': target_obj_class,
                    'target_loc_class': target_loc_class 
                    # 'object_coords': position_objects,
                    # 'mask_position_objects': mask
                }
            )
            return current_obs, (images[0], graph, graph_viz)

        else:
            obs = self.env.get_observations(char_index=1)
            class2id = {node['class_name']: node['id'] for node in obs['nodes']}
            category2id = {node['category']: node['id'] for node in obs['nodes']}

            ## filter graph
            obj_ids = []
            # class_types = ['character', 'kitchentable','coffeetable', 'kitchencounter', 'kitchencabinets', 'cabinet', 'bathroomcabinet', 'bookshelf', 'toilet', 'microwave', 'dishwahser', 'oven']
            class_types = ['character', 'kitchentable']
            for predicate in self.goal_spec:
                elements = predicate.split('_')
                class_types += list(elements[1:])
            if self.level == 0: # single room level
                obj_ids = [node['id'] for node in obs['nodes'] if str(node['id']) in class_types or \
                                            node['class_name'] in class_types]
            else:
                obj_ids = [node['id'] for node in obs['nodes'] if str(node['id']) in class_types or \
                                            node['class_name'] in class_types or \
                                            node['category'] == 'Rooms']
            filtered_obs = {
                'nodes': [node for node in obs['nodes'] if node['id'] in obj_ids],
                'edges': [edge for edge in obs['edges'] if edge['from_id'] in obj_ids and edge['to_id'] in obj_ids] 
            }
            # print([(node['id'], node['class_name'])for node in filtered_obs['nodes']])
            # ipdb.set_trace()

            graph_inputs, graph_viz = self.graph_helper.build_graph(filtered_obs,
                                                                    character_id=self.my_agent_id, plot_graph=drawing)


            current_obs = graph_inputs
            current_obs['affordance_matrix'] = self.graph_helper.obj1_affordance
            self.visible_nodes = graph_viz[-1]
            return current_obs, (None, graph_viz)

    def print_action(self, system_agent_action, my_agent_action):
        self.actions['system_agent'].append(system_agent_action)
        self.actions['my_agent'].append(my_agent_action)

        system_agent_actions = self.actions['system_agent']
        my_agent_actions = self.actions['my_agent']
        num_steps = len(system_agent_actions)

        print('**************************************************************************')
        if self.num_agents>1:
            for i in range(num_steps):
                print('step %04d:\t|"system": %s \t\t\t\t\t\t |"my_agent": %s' % (i+1, system_agent_actions[i].ljust(30), my_agent_actions[i]))
        else:
            for i in range(num_steps):
                print('step %04d:\t|"system": %s' % (i+1, system_agent_actions[i]))

        print('**************************************************************************')
