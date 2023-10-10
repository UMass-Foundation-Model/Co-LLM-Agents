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
                script = file_info['instruction']  # 从文件中获取指令
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


exit()

os.system(f'kill -9 $(lsof -t -i:{8080})')

comm = comm_unity.UnityCommunication(file_name="../../executable/linux_exec.v2.3.0.x86_64", port="8080", x_display= '99', no_graphics=False)
for record_step_counter in range(1, file_num+1):
# for record_step_counter in range(1, 2):
    with open(os.path.join(graph_save_dir, 'file_{}.json'.format(record_step_counter)), 'r') as f:
        file_info = json.load(f)

    graph = file_info['graph']  # 从文件中获取图数据
    script = file_info['instruction']  # 从文件中获取指令
    time_elapsed = file_info['time']  # 从文件中获取经过的时间
    total_completed = file_info['predicates']  # 从文件中获取已完成的谓词
    env_id = env_task_set[task_num]['env_id']

    comm.reset(env_id)


    # print(graph)
    agentnodes = [i for i in graph['nodes'] if i['id'] in [1,2]]
    agentedges = [i for i in graph['edges'] if i['from_id'] in [1,2] or i['to_id'] in [1,2]]

    # graph['nodes'] = [i for i in graph['nodes'] if i['id'] not in [1,2]]
    # graph['edges'] = [i for i in graph['edges'] if i['from_id'] not in [1,2] and i['to_id'] not in [1,2]]
    comm.expand_scene(graph)
    # comm.add_character()
    print(agentnodes)
    i,j = agentnodes
    assert i['id'] == 1
    assert j['id'] == 2

    print([i for i in comm.environment_graph()[1]['nodes'] if i['id'] in [1,2]])
    # comm.add_camera(position = [0, 1.8, 0.15], rotation = [30, 0, 90], field_view = 90, name = 'up_camera')
    # num_static_cameras = comm.camera_count()[1]
    # comm.add_character('Chars/'+i['prefab_name'], position=i['obj_transform']['position'])
    # comm.add_character('Chars/'+j['prefab_name'], position=j['obj_transform']['position'])
    camera_ids = [83]
    s, bgr_images = comm.camera_image(camera_ids, mode='normal', image_width=1024, image_height=1024)
    for i in range(len(camera_ids)):
        rot_bgr = bgr_images[i]
        save_dir = os.path.join('saved_images', args.graph_save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        umat_bgr = cv2.UMat(rot_bgr)
        cv2.putText(umat_bgr, str(script), (20, 1024-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(save_dir, f"{record_step_counter:03}_img.png"), umat_bgr.get())
comm.close()