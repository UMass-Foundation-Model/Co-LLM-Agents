import pickle
import pdb
import sys
import os
import random
import json
import numpy as np
import copy
import argparse
import pickle as pkl
import ipdb


curr_dir = os.path.dirname(os.path.abspath(__file__))
home_path = '../../'
sys.path.append(f'{curr_dir}/../../virtualhome')

from simulation.unity_simulator import comm_unity as comm_unity
from init_goal_setter.init_goal_base import SetInitialGoal
from init_goal_setter.tasks import Task

parser = argparse.ArgumentParser()
parser.add_argument('--num-per-task', type=int, default=10, help='Maximum #episodes/task')
parser.add_argument('--seed', type=int, default=15, help='Seed for the apartments')

parser.add_argument('--task', type=str, default='setup_table', help='Task name')
parser.add_argument('--demo-id', type=int, default=0, help='demo index')
parser.add_argument('--port-number', type=int, default=8290, help='port')
parser.add_argument('--logging', action='store_true', default=False, help='Use unity editor')
parser.add_argument('--use-editor', action='store_true', default=False, help='Use unity editor')
parser.add_argument('--display', type=str, default='2', help='display')
parser.add_argument('--exec_file', type=str, default='../executable/linux_exec.v2.3.0.x86_64', help='Path to the executable')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed == 0:
        rand = random.Random()
    else:
        rand = random.Random(args.seed)

    with open(f'{curr_dir}/data/init_pool.json') as file:
        init_pool = json.load(file)

    file_split = f'{curr_dir}/../dataset/watch_scenes_split.json'

    with open(file_split, 'r') as f:
        content_split = json.load(f)

    # Predicates train and test
    cache_files = []
    predicates = {'train': [], 'test': []}


    for elem in content_split['test']:
        # Create a has pred str
        pred_dict = elem['pred_dict']
        pred_str = ','.join(sorted([x+'.'+str(y) for x,y in pred_dict.items()]))

        if pred_str not in cache_files:
            cache_files.append(elem['pred_str'])
        else:
            continue
        predicates['test'].append((elem['pred_dict'], elem['task_name'], pred_str))

    simulator_args={
                   'file_name': args.exec_file,
                    'x_display': 0,
                    'logging': args.logging,
                    'no_graphics': True,
                }

    if args.use_editor: 
        comm = comm_unity.UnityCommunication()
    else:
        comm = comm_unity.UnityCommunication(port=str(args.port_number), **simulator_args)
    comm.reset()
    s, graph = comm.environment_graph()
    

    
    ## -------------------------------------------------------------
    ## step3 load object size
    with open(f'{curr_dir}/data/class_name_size.json', 'r') as file:
        class_name_size = json.load(file)

    ## -------------------------------------------------------------
    ## gen graph
    ## -------------------------------------------------------------
    task_names = {  1: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    2: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    3: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    4: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    5: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food"],
                    6: ["setup_table", "clean_table", "put_fridge", "prepare_food", "read_book", "watch_tv"],
                    7: ["setup_table", "clean_table", "put_dishwasher", "unload_dishwasher", "put_fridge", "prepare_food", "read_book", "watch_tv"]}
                    

    success_init_graph = []

    apartment_list = [3, 6] 

    task_counts = {"setup_table": 0, 
                   "put_dishwasher": 0, 
                   "put_fridge": 0, 
                   "prepare_food": 0,
                   "read_book": 0}

    test_set = []

    task_id = 0


    for predicates_dict, task_name, pred_str in predicates['test']:
        demo_goals = copy.deepcopy(predicates_dict)
        if task_counts[task_name] >= 10:
            continue
        #print('task_name:', task_name)
        #print('goals:', demo_goals)
        filter_goals = {}
        for goal_description in demo_goals.keys():
            if 'ds_' not in goal_description and 'sit_' not in goal_description and 'glass' not in goal_description:
                filter_goals[goal_description] = demo_goals[goal_description]

        #print('filter_goals:', filter_goals)

        task_goal = {}
        for i in range(2):
            task_goal[i] = filter_goals

        goal_class = {}
        for predicate, count in task_goal[0].items():
            elements = predicate.split('_')
            if elements[2].isdigit():
                ipdb.set_trace()
                new_predicate = '{}_{}_{}'.format(elements[0], elements[1], id2node[int(elements[2])]['class_name'])
                location_name = id2node[int(elements[2])]['class_name']
            else:
                new_predicate = predicate
            goal_class[new_predicate] = count

        num_test = 100
        count_success = 0

        for i in range(num_test):
            # Select apartments that allow the task
            apt_list = [capt for capt in apartment_list if task_name in task_names[capt+1]]
            assert(len(apt_list) > 0)
            apartment = rand.choice(apt_list)
            comm.reset(apartment)
            s, original_graph = comm.environment_graph()
            graph = copy.deepcopy(original_graph)

            with open(f'{curr_dir}/data/object_info%s.json'%(apartment+1), 'r') as file:
                obj_position = json.load(file)

            for obj, pos_list in obj_position.items():
                if obj in ['book', 'remotecontrol']:
                    positions = [pos for pos in pos_list if \
                    pos[0] == 'INSIDE' and pos[1] in ['kitchencabinet', 'cabinet'] or \
                    pos[0] == 'ON' and pos[1] in \
                    (['cabinet', 'bench', 'nightstand'] + ([] if apartment == 2 else ['kitchentable']))]
                elif obj == 'remotecontrol':
                     positions = [pos for pos in pos_list if pos[0] == 'ON' and pos[1] in \
                    ['tvstand']]
                else:
                    positions = [pos for pos in pos_list if \
                    pos[0] == 'INSIDE' and pos[1] in ['fridge', 'kitchencabinets', 'cabinet', 'microwave', 'dishwasher', 'stove'] or \
                    pos[0] == 'ON' and pos[1] in \
                        (['cabinet', 'coffeetable', 'bench']+ ([] if apartment == 2 else ['kitchentable']))]
                obj_position[obj] = positions
            #print(obj_position['cutleryfork'])


            print('------------------------------------------------------------------------------')
            print('testing %d: %s' % (i, task_name))
            print('------------------------------------------------------------------------------')
            
            ## -------------------------------------------------------------
            ## setup goal based on currect environment
            ## -------------------------------------------------------------
            set_init_goal = SetInitialGoal(obj_position, class_name_size, init_pool, task_name, same_room=False, goal_template=demo_goals, rand=rand)
            # pdb.set_trace()
            init_graph, env_goal, success_expand = getattr(Task, task_name)(set_init_goal, graph)

            #print(task_goal)
            total_count = 0
            if success_expand:
                
                success, message = comm.expand_scene(init_graph)
            #    print('----------------------------------------------------------------------')
            #    print(task_name, success, message, set_init_goal.num_other_obj)
                # print(env_goal)
                if not success:
                    goal_objs = []
                    goal_names = []
                    for k,goals in env_goal.items():
                        goal_objs += [int(list(goal.keys())[0].split('_')[-1]) for goal in goals if list(goal.keys())[0].split('_')[-1] not in ['book', 'remotecontrol']]
                        goal_names += [list(goal.keys())[0].split('_')[1] for goal in goals]
                    
                    obj_names = [obj.split('.')[0] for obj in message['unplaced']]
                    obj_ids = [int(obj.split('.')[1]) for obj in message['unplaced']]
                    id2node = {node['id']: node for node in init_graph['nodes']}
            #        for obj_id in obj_ids:
            #            print([id2node[edge['to_id']]['class_name'] for edge in init_graph['edges'] if edge['from_id'] == obj_id])

                    if task_name!='read_book' and task_name!='watch_tv':
                        intersection = set(obj_names) & set(goal_names)
                    else:
                        intersection = set(obj_ids) & set(goal_objs)
                    
                    ## goal objects cannot be placed
                    if len(intersection)!=0:
                        success2 = False
                    else:
                        init_graph = set_init_goal.remove_obj(init_graph, obj_ids)
                        success2, message2 = comm.expand_scene(init_graph, transfer_transform=False)
                        success = True
                
                else:
                    success2 = True
                
                
                if success2 and success:

                    success = set_init_goal.check_goal_achievable(init_graph, comm, env_goal, apartment)

                    if success:
                        comm.reset(apartment)

                        init_graph0 = copy.deepcopy(init_graph)
                        s, m = comm.expand_scene(init_graph, transfer_transform=False)
                        if not s:
                            ipdb.set_trace()
                        s, init_graph = comm.environment_graph()
                        #print('final s:', s)
                        if s:
                            for subgoal in env_goal[task_name]:
                                for k, v in subgoal.items():
                                    elements = k.split('_')
                                    if len(elements) == 4:
                                        obj_class_name = elements[1]
                                        ids = [node['id'] for node in init_graph['nodes'] if node['class_name'] == obj_class_name]
                        #                print(obj_class_name, v, ids)
                                        if len(ids) < v:
                        #                    print(obj_class_name, v, ids)

                                            ipdb.set_trace()
                                            s = 0
                                            break

                            count_success += s

                        if s:
                            cur_goal_spec = {}
                            total_goal_spec = {}
                            see_container_list = [
                                'bathroomcabinet',
                                'kitchencabinet',
                                'cabinet',
                                'fridge',
                                'stove',
                                # 'kitchencounterdrawer',
			                    # 'coffeepot',
                                # 'dishwasher',
                                'microwave',
                            ]
                            items_list = list(task_goal[0].items())
                            random.shuffle(items_list)
                            for predicate, count in items_list:
                                elements = predicate.split('_')
                                goal_obj_name = elements[1]
                                goal_obj_ids = [node['id'] for node in init_graph['nodes'] if node['class_name'] == goal_obj_name]
                                max_count_1 = len(goal_obj_ids)
                                id2inside = {edge['from_id']: edge['to_id'] for edge in init_graph['edges'] if edge['relation_type'] == 'INSIDE'}
                                id2name = {node['id']: node['class_name'] for node in init_graph['nodes']}
                                can_see_container_ids = [node['id'] for node in init_graph['nodes'] if node['class_name'] in see_container_list or node['category'] == 'Rooms']
                                can_reach_goal_obj_ids = [x for x in goal_obj_ids if id2inside[x] in can_see_container_ids]
                                max_count = len(can_reach_goal_obj_ids)
                                count = min(max(1, random.randint(max_count - 2, max_count)), max_count)
                                count = min(5 - total_count, count)
                                total_count += count
                                task_goal[0][predicate] = count
                            #    print('predicate:', predicate, 'count:', count, 'max_count:', max_count)
                            #    assert count <= max_count
                            #    assert max_count_1 == max_count
                            #    import time
                            #    time.sleep(1010)
                                if elements[2].isdigit():
                                    location_id = list([node['id'] for node in init_graph['nodes'] if node['class_name'] == location_name])[0]
                                    new_predicate = '{}_{}_{}'.format(elements[0], elements[1], location_id)
                                    cur_goal_spec[new_predicate] = count
                                else:
                                    if elements[2] == 'character':
                                        location_id = 1
                                    else:
                                        location_name = elements[2]
                                        location_id = list([node['id'] for node in init_graph['nodes'] if
                                                            node['class_name'] == location_name])[0]
                                    new_predicate = '{}_{}_{}'.format(elements[0], elements[1], location_id)
                                    cur_goal_spec[new_predicate] = count
                                    total_goal_spec[new_predicate] = max_count_1

                            filter_cur_goal_spec = {}
                            filter_total_goal_spec = {}
                            for predicate, count in cur_goal_spec.items():
                                if count > 0:
                                    filter_cur_goal_spec[predicate] = count
                                    filter_total_goal_spec[predicate] = total_goal_spec[predicate]
                                    assert filter_cur_goal_spec[predicate] <= filter_total_goal_spec[predicate]
                            cur_goal_spec = filter_cur_goal_spec
                            total_goal_spec = filter_total_goal_spec
                            print('cur_goal_spec:', cur_goal_spec)
                            print('total_goal_spec:', total_goal_spec)
                            if total_count >= 3:
                                test_set.append({'task_id': task_id, 
                                      'task_name': task_name, 
                                      'env_id': apartment, 
                                      'init_graph': init_graph, 
                                      'task_goal': {0: cur_goal_spec, 1: cur_goal_spec},
                                      'total_goal': total_goal_spec,
                                      'goal_class': goal_class,
                                      'level': 1, 
                                      'init_rooms': rand.sample(['kitchen', 'bedroom', 'livingroom', 'bathroom'], 2),
                                      'pred_str': pred_str})
                                task_id += 1
                        #    print('task_name:', test_set[-1]['task_name'])
                        #    print('task_goal:', test_set[-1]['task_goal'])
                            else: count_success = 0
                            
            print('apartment: %d: success %d over %d (total: %d), goal count = %d' % (apartment, count_success, i+1, num_test, total_count))

            if count_success>=1:
                task_counts[task_name] += 1
                break
    #ipdb.set_trace()
    print('length of test set:', len(test_set))
    pickle.dump(test_set, open(f'{curr_dir}/../dataset/test_env_set_help.pik', 'wb'))