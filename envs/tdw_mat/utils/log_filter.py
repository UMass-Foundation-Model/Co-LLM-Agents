import os
import re
from numpy import array
from collections import defaultdict
import pandas as pd
import argparse
# Extract LLM prompts and outputs from log files.

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="results")
parser.add_argument("--log_path", type=str, default="output.log")
parser.add_argument("--output_path", type=str, default="LLM_data.csv")
parser.add_argument("--type", type=str, default="nips")
args = parser.parse_args()

log_dir = args.log_dir
log_path = os.path.join(log_dir, args.log_path)
output_path = os.path.join(log_dir, args.output_path)

# if not os.path.exists(output_path):
#     with open(logger_path, 'r') as f:
#         log = f.readlines()
#         for l in log:
#             if 'cot_output' in l or ('Episode: ' in l and '/12' in l):
#                 with open(output_path, 'a') as ff:
#                     ff.write(l)
LLM_data = defaultdict(list)

with open(log_path, 'r') as f:
    log = f.readlines()
    episode = 1
    for l in log:
        len_text = len(l)
        if len_text > 100:
            len_text = 100
        if 'DEBUG' not in l[:len_text]:
            if 'Episode: ' in l and '/12' in l:
                episode = int(re.search(r'Episode: (\d+)/\d+', l).group(1))
            continue
            # with open(output_path, 'a') as ff:
            #     ff.write(l)
        else:
            if "'LLM'" not in l:
                continue
            pure_text = l[l.find('{'):]
            llm_output = eval(pure_text)['LLM']
            if args.type == 'nips':
                agent_name = llm_output['prompts'][4:llm_output['prompts'].find('. ')]
                # print(agent_name)
                LLM_data['episode'].append(episode)
                LLM_data['agent'].append(agent_name)

                LLM_data['prompt'].append(llm_output['prompts'])
                LLM_data['output_plan_stage_1'].append(llm_output['cot_outputs'][0])
                LLM_data['output_plan_stage_2'].append(llm_output['outputs'][0])
                # print(llm_output)
                LLM_data['output_comm'].append(llm_output['message_generator_outputs'][0] if 'message_generator_outputs' in llm_output else "")
                LLM_data['output_parse_results'].append(llm_output['plan'])
            else:
                prompt = llm_output['prompt']
                output = llm_output['output'][0]
                agent_name = prompt[4:prompt.find('. ')]
                # print(agent_name)
                LLM_data['episode'].append(episode)
                LLM_data['agent'].append(agent_name)
                LLM_data['prompt_plan'].append(prompt)
                # print(prompt)
                LLM_data['output_plan_stage_1'].append(re.search(r"Let's think step by step\.(.*?) So the answer is option", prompt, re.S).group(1))
                LLM_data['output_plan_stage_2'].append(output)
                # print(llm_output)
                LLM_data['prompt_comm'].append(llm_output['message_generator_prompt'] if 'message_generator_prompt' in llm_output else "")
                LLM_data['output_comm'].append(llm_output['message_generator_outputs'][0] if 'message_generator_outputs' in llm_output else "")
                LLM_data['parse_exception'].append(llm_output['parse_exception'])
                LLM_data['output_parse_results'].append(llm_output['plan'])
    df =pd.DataFrame(LLM_data)
    # print(df)
    df.to_csv(output_path)
            # if 'cot_outputs' not in llm_output:
            #     cot_out = ''
            # else:
            #     cot_out = llm_output['cot_outputs'][0]
            # if 'plan' not in llm_output:
            #     plan = 'None'
            # else:
            #     plan = llm_output['plan']
            # find_frame_str = llm_output['prompts'][: llm_output['prompts'].find('3000', llm_output['prompts'].find('3000') + 1)]
            # find_frame_str = find_frame_str[find_frame_str.find(' ', -6) + 1: -1]
            # with open(output_path, 'a') as ff:
            #     ff.write('frame: ' + str(find_frame_str) + ', Agent: ' + agent_name + ', cot_output: ' + str(cot_out) + ', plan: ' + str(plan) + '\n')
