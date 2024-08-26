import json,os,sys
import numpy as np
import cv2
sys.path.append(os.path.join('..','..','virtualhome', 'simulation'))
import argparse
import glob
from unity_simulator import comm_unity
os.system("Xvfb :99 & export DISPLAY=:99")
import time
import pickle
time.sleep(3)
parser = argparse.ArgumentParser()
parser.add_argument('--graph_save_dir', default = 'record_graph/LLM_comm/task_0/trial_0/time.05.09.2023-12.51.31', type=str, help="The dir that save the graph data")
args = parser.parse_args()

graph_save_dir = args.graph_save_dir
task_num = 0




env_task_set = pickle.load(open("../dataset/test_env_set_help.pik", 'rb'))


import os
from collections import defaultdict

root_dir = 'record_graph'
result = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))

for dirpath, dirnames, filenames in os.walk(root_dir):
    parts = dirpath.split('/')
    if len(parts) == 5 and True and parts[2].startswith('task_') and parts[3].startswith('trial_') and parts[4].startswith('time.'):
        method = parts[1]
        task = parts[2]
        trial = parts[3]
        time = parts[4]
        if time > result[method][task][trial]:
            result[method][task][trial] = time

allpaths = []
for method, tasks in result.items():
    for task, trials in tasks.items():
        for trial, time in trials.items():
            onepath = f'{root_dir}/{method}/{task}/{trial}/{time}'
            allpaths.append(onepath)
            filename_template = 'file_*.json'
            file_list = glob.glob(os.path.join(onepath, filename_template))
            file_num = len(file_list)# graph_save_dir = 'record_graph/debugtest/task_0/time.05.06.2023-06.17.07'
            comm = [0,0]
            for record_step_counter in range(1, file_num+1):
                with open(os.path.join(onepath, 'file_{}.json'.format(record_step_counter)), 'r') as f:
                    file_info = json.load(f)
                script = file_info['instruction']
                try:
                    comm[0] += script['0'].startswith('[send_message] ')
                except:
                    pass
                try:
                    comm[1] += script['1'].startswith('[send_message] ')
                except:
                    pass
            
            print(trial, task, method, comm, file_num)
print(allpaths)