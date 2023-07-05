import os
from numpy import array
# open a log, extract llm cot_output and plan.
logger_path = 'results/4002/output.log'
output_path = 'results/4002/output.txt'
output_filter_path = 'results/4002/output_filter.txt'

if not os.path.exists(output_path):
    with open(logger_path, 'r') as f:
        log = f.readlines()
        for l in log:
            if 'cot_output' in l or ('Episode: ' in l and '/12' in l):
                with open(output_path, 'a') as ff:
                    ff.write(l)

if not os.path.exists(output_filter_path):
    with open(output_path, 'r') as f:
        log = f.readlines()
        for l in log:
            len_text = len(l)
            if len_text > 100:
                len_text = 100
            if 'DEBUG' not in l[:len_text]:
#                continue
                with open(output_filter_path, 'a') as ff:
                    ff.write(l)
            else:
                pure_text = l[l.find('{'):]
#                print('pure_text:', pure_text)
                agent_name = eval(pure_text)['LLM']['prompts'][4 :eval(pure_text)['LLM']['prompts'].find('. ')]
                print(agent_name)
                llm_output = eval(pure_text)['LLM']
                if 'cot_outputs' not in llm_output:
                    cot_out = '[]'
                else:
                    cot_out = llm_output['cot_outputs']
                if 'plan' not in llm_output:
                    plan = 'None'
                else:
                    plan = llm_output['plan']
                find_frame_str = llm_output['prompts'][: llm_output['prompts'].find('3000', llm_output['prompts'].find('3000') + 1)]
                find_frame_str = find_frame_str[find_frame_str.find(' ', -6) + 1: -1]
                with open(output_filter_path, 'a') as ff:
                    ff.write('frame: ' + str(find_frame_str) + ', Agent: ' + agent_name + ', cot_output: ' + str(cot_out) + ', plan: ' + str(plan) + '\n')
    