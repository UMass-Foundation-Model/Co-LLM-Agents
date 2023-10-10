import sys
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{curr_dir}/..')
import ipdb
import pickle
import json
import random
import numpy as np
from pathlib import Path

from envs.unity_environment import UnityEnvironment
from agents import vision_LLM_agent
from arguments import get_args
from algos.arena_mp2 import ArenaMP


if __name__ == '__main__':
    args = get_args()
    env_task_set = pickle.load(open(args.dataset_path, 'rb'))
    # with open("test_env.json", "w") as f:
    #     json.dump(env_task_set, f, indent=4)

    args.record_dir = f'../test_results/{args.mode}' # set the record_dir right!
    Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    if "image" in args.obs_type:
        os.system("Xvfb :98 & export DISPLAY=:98")
        import time
        time.sleep(3) # ensure Xvfb is open
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        executable_args = {
                        'file_name': args.executable_file,
                        'x_display': '98',
                        'no_graphics': False,
                        'timeout_wait': 5000,
        }
    else:
        executable_args = {
                        'file_name': args.executable_file,
                        'no_graphics': True,
        }

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))
    episode_ids = sorted(episode_ids)
    num_tries = args.num_runs
    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]


    def env_fn(env_id):
        return UnityEnvironment(num_agents=2,
                               max_episode_length=args.max_episode_length,
                               port_id=env_id,
                               env_task_set=env_task_set,
                               agent_goals=['LLM', 'LLM'],
                               observation_types=[args.obs_type, args.obs_type],
                               use_editor=args.use_editor,
                               executable_args=executable_args,
                               base_port=args.base_port)

    args_agent1 = {
        'agent_id': 1,
        'char_index': 0,
        'args': args,
    }
    args_agent2 = {
        'agent_id': 2,
        'char_index': 1,
        'args': args,
    }

    agents = [lambda x, y: vision_LLM_agent(**args_agent1), lambda x, y: vision_LLM_agent(**args_agent2)]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents, args.record_dir, args.debug)

    # copy the code below to record results
    if args.num_per_task != 10:
        test_episodes = args.test_task
    else:
        test_episodes = episode_ids
    for iter_id in range(num_tries):
        steps_list, failed_tasks = [], []
        if not os.path.isfile(args.record_dir + '/results.pik'):
            test_results = {}
        else:
            test_results = pickle.load(open(args.record_dir + '/results.pik', 'rb'))

        current_tried = iter_id

        for episode_id in test_episodes:
            curr_log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(
                env_task_set[episode_id]['task_id'],
                env_task_set[episode_id]['task_name'],
                iter_id)

            if os.path.isfile(curr_log_file_name):
                with open(curr_log_file_name, 'rb') as fd:
                    file_data = pickle.load(fd)
                S[episode_id].append(file_data['finished'])
                L[episode_id].append(max(len(file_data['action'][0]), len(file_data['action'][1])))

                test_results[episode_id] = {'S': S[episode_id],
                                            'L': L[episode_id]}
                continue

            print('episode:', episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2

            is_finished = 0
            steps = 250
            # try:
            arena.reset(episode_id)
            success, steps, saved_info = arena.run()
            print('-------------------------------------')
            print('success' if success else 'failure')
            print('steps:', steps)
            print('-------------------------------------')
            if not success:
                failed_tasks.append(episode_id)
            else:
                steps_list.append(steps)
            is_finished = 1 if success else 0
            log_file_name = args.record_dir + '/logs_agent_{}_{}_{}.pik'.format(saved_info['task_id'],
                                                                                saved_info['task_name'],
                                                                                current_tried)

            if len(saved_info['obs']) > 0:
                pickle.dump(saved_info, open(log_file_name, 'wb'))
            else:
                with open(log_file_name, 'w+') as f:
                    f.write(json.dumps(saved_info, indent=4))
            # except:
            #     # ipdb.set_trace()
            #     arena.reset_env()

            S[episode_id].append(is_finished)
            L[episode_id].append(steps)

            test_results[episode_id] = {'S': S[episode_id],
                                        'L': L[episode_id]}

        print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
        print('failed_tasks:', failed_tasks)
        pickle.dump(test_results, open(args.record_dir + '/results.pik', 'wb'))

