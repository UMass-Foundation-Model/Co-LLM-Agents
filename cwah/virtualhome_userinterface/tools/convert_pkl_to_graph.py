import pickle as pkl
import os
import ipdb
import json


def convert_plot(out_path = '../record_graph/planner/', file_input='../data_input/log_agent_0_read_book.pik'):

    out_path_all = out_path + file_input.split('/')[-1].replace('.pik', '')
    if not os.path.isdir(out_path_all):
        os.makedirs(out_path_all)
    with open(file_input, 'rb') as f:
            content = pkl.load(f)

    obs0 = content['obs'][0]
    obs_char = [node for node in obs0 if node['id'] == 1][0]

    with open('{}/init_graph.json'.format(out_path_all), 'w+') as f:
            graph_data = {'graph': content['init_unity_graph']}
            graph_data['graph']['nodes'].append(obs_char)
            json.dump(graph_data, f)

    all_ids = [1] + [node['id'] for node in graph_data['graph']['nodes']]
    def convert_data(node):
            # print(node)
            new_node = {name: node[name] for name in ['id', 'class_name', 'states', 'bounding_box']}
            if new_node['id'] == 1:
                    new_node['bounding_box']['size'] = [0.75, 0.75, 0.75]
            return new_node


    # ipdb.set_trace()
    # First time an id is used
    first_time_id = {}
    for it in range(len(content['action'][0])):
            ids_obs = [(node['id'], node) for node in content['obs'][it]]
            for idi, node in ids_obs:
                    if idi not in first_time_id:
                            first_time_id[idi] = node


    last_id = {}

    for it in range(len(content['action'][0])):
            instr = content['action'][0][it]
            info = {
                    'instruction': [instr]
            }
            obs = content['obs'][it]
            graph = {
                    'nodes': obs,
                    'edges': []
            }
            graph['nodes'] = [convert_data(node) for node in graph['nodes']]
            
            missing_ids = set(all_ids) - set([node['id'] for node in graph['nodes']])
            for missing_id in missing_ids:
                    if missing_id in last_id:
                            graph['nodes'].append(last_id[missing_id])
                    else:
                            if missing_id in first_time_id:
                                    graph['nodes'].append(convert_data(first_time_id[missing_id]))

            # keep track of last ids
            for node in graph['nodes']:
                    last_id[node['id']] = node	

            info['graph'] = graph
            with open('{}/file_{}.json'.format(out_path_all, it), 'w+') as f:
                    json.dump(info, f)

files = [
    'logs_agent_0_read_book_0.pik'
    , 'logs_agent_1_read_book_0.pik'
    , 'logs_agent_20_put_dishwasher_0.pik'
    , 'logs_agent_21_put_dishwasher_0.pik'
    , 'logs_agent_40_prepare_food_0.pik'
    , 'logs_agent_41_prepare_food_0.pik'
    , 'logs_agent_60_put_fridge_0.pik'
    , 'logs_agent_61_put_fridge_0.pik'
    , 'logs_agent_80_setup_table_0.pik'
    , 'logs_agent_81_setup_table_0.pik'
]
for file_name in files:
    path_init = '../../MultiAgent/challenge/vh_multiagent_models/record_scratch/rec_good_test/multiAlice_env_task_set_20_check_neurips_test/'
    convert_plot(file_input=path_init + file_name)
