import random
import pdb
import copy
import time

class Arena:
    def __init__(self, agent_types, environment):
        self.agents = []
        for agent_type in agent_types:
            self.agents.append(agent_type)
        self.num_agents = len(agent_types)
        self.env = environment

    def reset(self, task_id=None):
        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id)
        for it, agent in enumerate(self.agents):
            if agent.agent_type == 'MCTS':
                agent.reset(self.env.python_graph, self.env.task_goal, seed=it)
            else:
                agent.reset(self.env.python_graph)

    def get_actions(self, obs, action_space=None):
        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}
        for it, agent in enumerate(self.agents):
            if agent.agent_type == 'MCTS':
                opponent_subgoal = None
                if agent.recursive:
                    opponent_subgoal = self.agents[1 - it].last_subgoal
                dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.task_goal[it] if it == 0 else self.task_goal[it], opponent_subgoal)
            elif agent.agent_type == 'RL':

                dict_actions[it], dict_info[it] = agent.get_action(obs[it], self.env.goal_spec if it == 0 else self.task_goal[it], action_space_ids=action_space[it])
        return dict_actions, dict_info

    def rollout(self, logging=False, record=False):
        t1 = time.time()
        self.reset()
        t2 = time.time()
        t_reset = t2 - t1
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0
        info_rollout = {}
        entropy_action, entropy_object = [], []
        observation_space, action_space = [], []

        info_rollout['step_info'] = []
        info_rollout['script'] = []

        rollout_agent = {}

        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            if agent.agent_type == 'RL':
                rollout_agent[agent_id] = []

        if logging:
            init_graph = self.env.get_graph()
            pred = self.env.goal_spec
            goal_class = list(pred.keys())[0].split('_')[1]
            id2node = {node['id']: node for node in init_graph['nodes']}
            info_goals = []
            info_goals.append([node for node in init_graph['nodes'] if node['class_name'] == goal_class])
            ids_target = [node['id'] for node in init_graph['nodes'] if node['class_name'] == goal_class]
            info_goals.append([(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type'],
                                edge['from_id']) for edge in init_graph['edges'] if edge['from_id'] in ids_target])
            info_rollout['target'] = [pred, info_goals]

        while not done and nb_steps < self.max_episode_length:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step()
            if logging:
                node_id = [node['bounding_box'] for node in obs[0]['nodes'] if node['id'] == 1][0]
                edges_char = [(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type']) for edge in init_graph['edges'] if edge['from_id'] == 1]

                info_rollout['step_info'].append((node_id, edges_char))
                info_rollout['script'].append(agent_actions[0])

            nb_steps += 1
            for agent_index in agent_info.keys():
                # currently single reward for both agents
                c_r_all[agent_index] += reward
                # action_dict[agent_index] = agent_info[agent_index]['action']

            entropy_action.append(-((agent_info[0]['probs'][0]+1e-9).log()*agent_info[0]['probs'][0]).sum().item())
            entropy_object.append(-((agent_info[0]['probs'][1]+1e-9).log()*agent_info[0]['probs'][1]).sum().item())
            observation_space.append(agent_info[0]['num_objects'])
            action_space.append(agent_info[0]['num_objects_action'])
            if record:
                actions.append(agent_actions)

            # append to memory
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].agent_type == 'RL':
                    state = agent_info[agent_id]['state_inputs']
                    policy = [log_prob.data for log_prob in agent_info[agent_id]['probs']]
                    action = agent_info[agent_id]['actions']
                    rewards = reward

                    rollout_agent[agent_id].append((state, policy, action, rewards, 1))


        t_steps = time.time() - t2
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        info_rollout['success'] = success_r_all[0]
        info_rollout['nsteps'] = nb_steps
        info_rollout['epsilon'] = self.agents[0].epsilon
        info_rollout['entropy'] = (entropy_action, entropy_object)
        info_rollout['observation_space'] = np.mean(observation_space)
        info_rollout['action_space'] = np.mean(action_space)
        info_rollout['t_reset'] = t_reset
        info_rollout['t_steps'] = t_steps

        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        info_rollout['success'] = success_r_all[0]
        info_rollout['nsteps'] = nb_steps
        info_rollout['epsilon'] = self.agents[0].epsilon
        info_rollout['entropy'] = (entropy_action, entropy_object)
        info_rollout['observation_space'] = np.mean(observation_space)
        info_rollout['action_space'] = np.mean(action_space)

        info_rollout['env_id'] = self.env.env_id
        info_rollout['goals'] = list(self.env.task_goal[0].keys())
        # padding
        # TODO: is this correct? Padding that is valid?
        while nb_steps < self.max_episode_length:
            nb_steps += 1
            for agent_id in range(self.num_agents):
                if self.agents[agent_id].agent_type == 'RL':
                    state = agent_info[agent_id]['state_inputs']
                    if 'edges' in obs.keys():
                        pdb.set_trace()
                    policy = [log_prob.data for log_prob in agent_info[agent_id]['probs']]
                    action = agent_info[agent_id]['actions']
                    rewards = reward
                    rollout_agent[agent_id].append((state, policy, action, 0, 0))

        return c_r_all, info_rollout, rollout_agent

    def step(self):
        obs = self.env.get_observations()
        action_space = self.env.get_action_space()
        dict_actions, dict_info = self.get_actions(obs, action_space)
        return self.env.step(dict_actions), dict_actions, dict_info


    def run(self, random_goal=False, pred_goal=None):
        """
        self.task_goal: goal inference
        self.env.task_goal: ground-truth goal
        """
        self.task_goal = copy.deepcopy(self.env.task_goal)
        if random_goal:
            for predicate in self.env.task_goal[0]:
                u = random.choice([0, 1, 2])
                self.task_goal[0][predicate] = u
                self.task_goal[1][predicate] = u
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        saved_info = {'task_id': self.env.task_id,
                      'env_id': self.env.env_id,
                      'task_name': self.env.task_name,
                      'gt_goals': self.env.task_goal[0],
                      'goals': self.task_goal[0],
                      'action': {0: [], 1: []},
                      'plan': {0: [], 1: []},
                      'subgoal': {0: [], 1: []},
                      # 'init_pos': {0: None, 1: None},
                      'finished': None,
                      'init_unity_graph': self.env.init_unity_graph,
                      'obs': []}
        success = False
        while True:
            (obs, reward, done, infos), actions, agent_info = self.step()
            success = infos['finished']
            for agent_id, action in actions.items():
                saved_info['action'][agent_id].append(action)
            for agent_id, info in agent_info.items():
                if 'plan' in info:
                    saved_info['plan'][agent_id].append(info['plan'][:3])
                if 'subgoal' in info:
                    saved_info['subgoal'][agent_id].append(info['subgoal'][:3])
                if 'obs' in info:
                    saved_info['obs'].append(info['obs'])
            if done:
                break
        saved_info['finished'] = success
        return success, self.env.steps, saved_info