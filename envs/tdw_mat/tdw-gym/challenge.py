import argparse
import os
import json
import gym
import time
import pickle
import logging
import sys

# add this dictionary to python env path:
base_path = os.getcwd()
sys.path.append(base_path)

from tdw_gym import TDW
from h_agent import H_agent
from lm_agent import lm_agent

gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw-gym.tdw_gym:TDW'
)

class Challenge:
    def __init__(self, logger, port, data_path, output_dir, number_of_agents = 2, max_frames = 3000, new_setting = True, screen_size = 256, data_prefix = 'dataset/nips_dataset/', gt_mask = True):
        self.env = gym.make("transport_challenge_MA", port = port, number_of_agents = number_of_agents, save_dir = output_dir, max_frames = max_frames, new_setting = new_setting, screen_size = screen_size, data_prefix = data_prefix, gt_mask = gt_mask)
        self.gt_mask = gt_mask
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.new_setting = new_setting
        if self.new_setting: self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        else: self.data = pickle.load(open(os.path.join(data_prefix, data_path), "rb"))
        self.logger.info("done")

    def submit(self, agents, logger, eval_episodes):
        total_finish = 0.0
        if eval_episodes == -1:
            num_eval_episodes = len(self.data)
        else:
            num_eval_episodes = eval_episodes
        
        start = time.time()
        for i in range(num_eval_episodes):
            start_time = time.time()
            if os.path.exists(os.path.join(self.output_dir, str(i), 'result_episode.json')):
                with open(os.path.join(self.output_dir, str(i), 'result_episode.json'), 'r') as f:
                    result = json.load(f)
                total_finish += result['finish'] / result['total']
                continue
            # The episode has been evaluated before
            if not os.path.exists(os.path.join(self.output_dir, str(i))):
                os.makedirs(os.path.join(self.output_dir, str(i)))
            self.logger.info('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data[i]}")
            state, info, env_api = self.env.reset(seed=self.data[i]['seed'], options=self.data[i], output_dir = os.path.join(self.output_dir, str(i)))
            for id, agent in enumerate(agents):
                if type(env_api) == list:
                    curr_api = env_api[id]
                else: curr_api = env_api
                if info['goal_description'] is not None:
                    if agent.agent_type == 'h_agent':
                        agent.reset(goal_objects = info['goal_description'], output_dir = os.path.join(self.output_dir, str(i)), env_api = curr_api, agent_color = info['agent_colors'][id], agent_id = id, gt_mask = self.gt_mask)
                    elif agent.agent_type == 'lm_agent':
                        agent.reset(obs = state[str(id)], goal_objects = info['goal_description'], output_dir = os.path.join(self.output_dir, str(i)), env_api = curr_api, rooms_name=info['rooms_name'])
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir = os.path.join(self.output_dir, str(i)))
            self.env.save_images(os.path.join(self.output_dir, str(i), 'Images'))
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            local_reward = 0.0
            while not done:
                step_num += 1
                actions = {}
                for agent_id, agent in enumerate(agents):
                    actions[str(agent_id)] = agent.act(state[str(agent_id)])
                state, reward, done, info = self.env.step(actions)
                self.env.save_images(os.path.join(self.output_dir, str(i), 'Images'))
                local_reward += reward
                local_finish = self.env.check_goal()
                self.logger.info(f"Executing step {step_num} for episode: {i}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}")
                if done:
                    break
            total_finish += local_finish[0] / local_finish[1]
            result = {
                "finish": local_finish[0],
                "total": local_finish[1],
            }
            with open(os.path.join(self.output_dir, str(i), 'result_episode.json'), 'w') as f:
                json.dump(result, f)
        avg_finish = total_finish / num_eval_episodes
        results = {
            "avg_finish": avg_finish
        }
        with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
            json.dump(results, f)
        self.logger.info(f'eval done, avg transport rate {avg_finish}')
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_finish

    def close(self):
        self.env.close()

def init_logs(output_dir, name = 'simple_example'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="dataset/arxiv_dataset_v3/")
    parser.add_argument("--port", default=1071, type=int)
    parser.add_argument("--agents", nargs= '+', type=str, default=("h_agent",))
    parser.add_argument("--eval_episodes", default=-1, type=int, help="how many episodes to evaluate on")
    parser.add_argument("--max_frames", default=3000, type=int, help="max frames per episode")
    parser.add_argument("--new_setting", default=True, action='store_true')
    parser.add_argument("--communication", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no_gt_mask", action='store_true')
    # LLM parameters
    parser.add_argument('--source', default='openai',
        choices=['huggingface', 'openai'],
        help='openai API or load huggingface models')
    parser.add_argument('--lm_id', default='gpt-3.5-turbo',
                        help='name for openai engine or huggingface model name/path')
    parser.add_argument('--prompt_template_path', default='LLM/prompt_single.csv',
                        help='path to prompt template file')
    parser.add_argument("--t", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument("--cot", action='store_true', help="use chain-of-thought prompt")
    parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")
    parser.add_argument("--screen_size", default=256, type=int)
    args = parser.parse_args()

    args.number_of_agents = len(args.agents)
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, str(args.run_id))
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)
    logger = init_logs(args.output_dir)

    challenge = Challenge(logger, args.port, args.data_path, args.output_dir, args.number_of_agents, args.max_frames, args.new_setting, data_prefix=args.data_prefix, gt_mask = not args.no_gt_mask)
    agents = []
    for i, agent in enumerate(args.agents):
        if agent == 'h_agent':
            agents.append(H_agent(i, logger, args.max_frames, args.output_dir))
        elif agent == 'lm_agent':
            agents.append(lm_agent(i, logger, args.max_frames, args, args.output_dir))
        else:
            pass
    try:
        challenge.submit(agents, logger, args.eval_episodes)
    finally:
        challenge.close()

if __name__ == "__main__":
    main()
