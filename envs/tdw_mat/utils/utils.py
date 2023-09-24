import json
import os
import re
import numpy as np
from numpy import array
from collections import defaultdict
import pandas as pd
import argparse

def LLM_filter(log_path, output_path, type, eval_episodes, eval_comm=False):
    LLM_data = defaultdict(list)
    comm = 0
    with open(log_path, 'r') as f:
        log = f.readlines()
        episode = -1
        for l in log:
            if "'LLM'" not in l:
                if 'Episode: ' in l and '/12' in l:
                    episode = int(re.search(r'Episode: (\d+)/\d+', l).group(1))
                if 'Episode ' in l:
                    episode = int(re.search(r'Episode (\d+)', l).group(1))
                continue
            if episode not in eval_episodes:
                continue
            pure_text = l[l.find('{'):]
            llm_output = eval(pure_text)['LLM']
            if type == 'nips':
                prompt = llm_output['prompts']
                agent_name = prompt[4:prompt.find('. ')]
                # print(agent_name)
                LLM_data['episode'].append(episode)
                LLM_data['agent'].append(agent_name)

                LLM_data['prompt_comm'].append(llm_output['message_generator_prompt'] if 'message_generator_prompt' in llm_output else "")
                LLM_data['output_comm'].append(llm_output['message_generator_outputs'][0] if 'message_generator_outputs' in llm_output else "")

                LLM_data['prompt_plan'].append(llm_output['prompts'])
                LLM_data['output_plan_stage_1'].append(llm_output['cot_outputs'][0])
                LLM_data['output_plan_stage_2'].append(llm_output['outputs'][0])
                # print(llm_output)
                LLM_data['output_parse_results'].append(llm_output['plan'])
                if 'send a message' in llm_output['plan']:
                    comm += 1
            else:
                prompt = llm_output['prompt_plan_stage_2']
                output = llm_output['output_plan_stage_2']
                agent_name = prompt[4:prompt.find('. ')]
                # print(agent_name)
                LLM_data['episode'].append(episode)
                LLM_data['agent'].append(agent_name)

                LLM_data['prompt_comm'].append(llm_output['prompt_comm'] if 'prompt_comm' in llm_output else "")
                LLM_data['output_comm'].append(llm_output['output_comm'][0] if 'output_comm' in llm_output else "")

                LLM_data['prompt_plan'].append(prompt)
                LLM_data['output_plan_stage_1'].append(re.search(r"Let's think step by step\.(.*?) Answer with only one best next action. So the answer is option", prompt, re.S).group(1))
                LLM_data['output_plan_stage_2'].append(output)
                # print(llm_output)
                LLM_data['parse_exception'].append(llm_output['parse_exception'])
                LLM_data['output_parse_results'].append(llm_output['plan'])
                if 'send a message' in llm_output['plan']:
                    comm += 1

        if eval_comm:
            print(comm)
        else:
            df =pd.DataFrame(LLM_data)
            # print(df)
            df[df['agent'] == 'Alice'].to_csv(output_path.replace('.csv', '_Alice.csv'))
            df[df['agent'] == 'Bob'].to_csv(output_path.replace('.csv', '_Bob.csv'))


def eval_EI(single_log_dir, log_dir, eval_episodes, result_path):
    fout = open(os.path.join(log_dir, result_path), 'w')
    print(f"episode\ttransport rate\tEI", file=fout)
    transport_rate = []
    EI = []
    for i, episode in enumerate(eval_episodes):
        result_list = []
        result_single_list = []
        result_file_list = os.listdir(log_dir)
        for runs in result_file_list:
            json_path = os.path.join(log_dir, runs, str(episode), 'result_episode.json')
            if os.path.exists(json_path):
                result = json.load(open(json_path, 'r'))
                result_list.append(result['finish'] / result['total'])
        single_result_file_list = os.listdir(single_log_dir)
        for runs in single_result_file_list:
            json_path = os.path.join(single_log_dir, runs, str(episode), 'result_episode.json')
            if os.path.exists(json_path):
                result = json.load(open(json_path, 'r'))
                result_single_list.append(result['finish'] / result['total'])
        tr = np.mean(result_list)
        tr_s = np.mean(result_single_list)
        ei = (tr - tr_s) / tr
        transport_rate.append(tr)
        EI.append(ei)
        print(f"{episode}\t{tr:.2f}\t{ei:.2f}", file=fout)
        print(f"{episode}\t{tr:.2f}\t{ei:.2f}")

    print(f"average\t{np.mean(transport_rate):.2f}\t{np.mean(EI):.2f}", file=fout)
    print(f"average\t{np.mean(transport_rate):.2f}\t{np.mean(EI):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLM_filter", action='store_true', help="Extract LLM prompts and outputs from log files.")
    parser.add_argument("--log_dir", type=str, default="results")
    parser.add_argument("--log_path", type=str, default="output.log")
    parser.add_argument("--result_path", type=str, default="results.tsv")
    parser.add_argument("--output_path", type=str, default="LLM_data.csv")
    parser.add_argument("--type", type=str, default="new", choices=['nips', 'new'])
    parser.add_argument("--eval_EI", action='store_true', help="calculate EI")
    parser.add_argument("--runs", nargs='+', default=(1, 2, 3, 4, 5), type=int)
    parser.add_argument("--eval_episodes", nargs='+', default=(-1,), type=int)
    parser.add_argument("--single_log_dir", type=str, default="results/LM-Llama-2-13b-hf")
    parser.add_argument("--eval_comm", action='store_true', help="calculate number of the comm")
    args = parser.parse_args()
    log_dir = args.log_dir
    eval_episodes = range(24)
    if args.eval_episodes[0] != -1:
        eval_episodes = args.eval_episodes
    if args.LLM_filter:
        log_path = os.path.join(log_dir, 'run_1', args.log_path)
        output_path = os.path.join(log_dir, args.output_path)
        LLM_filter(log_path, output_path, args.type, eval_episodes)
    if args.eval_EI:
        eval_EI(args.single_log_dir, log_dir, eval_episodes, args.result_path)
    if args.eval_comm:
        log_path = os.path.join(log_dir, args.log_path)
        output_path = os.path.join(log_dir, args.output_path)
        LLM_filter(log_path, output_path, args.type, eval_episodes, True)