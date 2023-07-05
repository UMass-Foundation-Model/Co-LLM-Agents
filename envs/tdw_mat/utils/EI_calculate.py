import os
import json
import numpy as np

result_path_single = ["eval_results/single_h_agent/1000", "eval_results/single_h_agent/1001", "eval_results/single_h_agent/1002", "eval_results/single_h_agent/1003", "eval_results/single_h_agent/1004"]
result_path_hp = ["eval_results/multi_h_agent/2000", "eval_results/multi_h_agent/2001", "eval_results/multi_h_agent/2002", "eval_results/multi_h_agent/2003", "eval_results/multi_h_agent/2004"]
#result_path_hp = ["eval_results/multi_h_llm_agent/3000"]
#result_path_hp = ["eval_results/multi_llm_agent/4000"]
single_tr, single_count = [0 for _ in range(12)], 0
for i in range(result_path_single.__len__()):
    for j in range(12):
        result_path = os.path.join(result_path_single[i], str(j), 'result_episode.json')
        with open(result_path, 'r') as f:
            result = json.load(f)
        single_tr[j] += result['finish'] / result['total']
    single_count += 1
for i in range(12):
    single_tr[i] /= single_count
    
hp_tr, hp_count = [0 for _ in range(12)], 0
for i in range(result_path_hp.__len__()):
    for j in range(12):
        result_path = os.path.join(result_path_hp[i], str(j), 'result_episode.json')
        with open(result_path, 'r') as f:
            result = json.load(f)
        hp_tr[j] += result['finish'] / result['total']
    hp_count += 1

for i in range(12): hp_tr[i] /= hp_count
    
EI = []
Rate = []
for i in range(12):
    EI.append((hp_tr[i] - single_tr[i]) / hp_tr[i])
    Rate.append(hp_tr[i])

print("Individual Rate:", Rate)
print("Rate:", np.mean(Rate))
print("Individual EI:", EI)
print("EI:", np.mean(EI))