import os
import json
import numpy as np

result_path_single = ["results/1008"]
result_path_hp = ["results/2001"]
total_episode = 24
#result_path_hp = ["eval_results/multi_h_llm_agent/3000"]
#result_path_hp = ["eval_results/multi_llm_agent/4000"]
single_tr, single_count = [0 for _ in range(total_episode)], 0
for i in range(result_path_single.__len__()):
    for j in range(total_episode):
        result_path = os.path.join(result_path_single[i], str(j), 'result_episode.json')
        with open(result_path, 'r') as f:
            result = json.load(f)
        single_tr[j] += result['finish'] / result['total']
    single_count += 1
for i in range(total_episode):
    single_tr[i] /= single_count

print(single_tr)

if type(result_path_hp[0]) == str:
    hp_tr, hp_count = [0 for _ in range(total_episode)], 0
    for i in range(result_path_hp.__len__()):
        for j in range(total_episode):
            result_path = os.path.join(result_path_hp[i], str(j), 'result_episode.json')
            with open(result_path, 'r') as f:
                result = json.load(f)
            hp_tr[j] += result['finish'] / result['total']
        hp_count += 1
    for i in range(total_episode): hp_tr[i] /= hp_count
else:
    hp_tr, hp_count = [0 for _ in range(total_episode)], 0
    for j in range(total_episode):
        hp_tr[j] = max(result_path_hp[j], 1)
    
EI = []
Rate = []
for i in range(total_episode):
    EI.append((hp_tr[i] - single_tr[i]) / hp_tr[i])
    Rate.append(hp_tr[i])

print("Individual Rate:", Rate)
print("Rate:", np.mean(Rate))
print("Individual EI:", EI)
print("EI:", np.mean(EI))