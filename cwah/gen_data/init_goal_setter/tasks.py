import random
import pdb
class Task:

    @staticmethod
    def setup_table(init_goal_manager, graph, start=True):
        ## setup table
        # max_num_table = 4
        # num_table = init_goal_manager.rand.randint(1, max_num_table)

        # table_ids = [node['id'] for node in graph['nodes'] if 'table' in node['class_name']]
        # init_goal_manager.remove_obj(graph, table_ids)
        # table_position_pool = init_goal_manager.obj_position['table']
        # init_goal_manager.add_obj(graph, 'table', num_table, table_position_pool)

        # table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_ids = [node['id'] for node in graph['nodes'] if ('kitchentable' in node['class_name'])]
        table_id = init_goal_manager.rand.choice(table_ids)

        ## remove objects on table
        id2node = {node['id']: node for node in graph['nodes']}
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if
                         (edge['to_id'] == table_id) and (edge['relation_type'] == 'ON') and \
                         id2node[edge['from_id']]['class_name'] in ['plate', 'cutleryfork', 'waterglass', 'wineglass',
                                                                    'book', 'poundcake', 'cutleryknife']]
        graph = init_goal_manager.remove_obj(graph, objs_on_table)

        # ## remove objects on kitchen counter
        # kitchencounter_ids = [node['id'] for node in graph['nodes'] if (node['class_name'] == 'kichencounter')]
        # for kichencounter_id in kitchencounter_ids:
        #     objs_on_kichencounter = [edge['from_id'] for edge in graph['edges'] if (edge['to_id']==kitchencounter_id) and (edge['relation_type']=='ON')]
        #     graph = init_goal_manager.remove_obj(graph, objs_on_kichencounter)

        # tem = [node for node in graph['nodes'] if node['id']==table_id]
        # pdb.set_trace()

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(table_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(table_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room, except_position=except_position_ids,
                                                                goal_obj=True)
            # print([node for node in graph['nodes'] if node['class_name'] == 'wineglass'])
            if not success:
                return None, None, False

        # pdb.set_trace()
        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                                          except_position=except_position_ids)

        ## get goal
        env_goal = {'setup_table': []}
        for k, v in init_goal_manager.goal.items():
            env_goal['setup_table'].append({'put_{}_on_{}'.format(k, table_id): v})

        return graph, env_goal, True


    @staticmethod
    def clean_table(init_goal_manager, graph, start=True):
        ## clean table
        # max_num_table = 4
        # num_table = init_goal_manager.rand.randint(1, max_num_table)

        # table_ids = [node['id'] for node in graph['nodes'] if 'table' in node['class_name']]
        # init_goal_manager.remove_obj(graph, table_ids)
        # table_position_pool = init_goal_manager.obj_position['table']
        # init_goal_manager.add_obj(graph, 'table', num_table, table_position_pool)

        # table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_ids = [node['id'] for node in graph['nodes'] if ('kitchentable' in node['class_name'])]
        table_id = init_goal_manager.rand.choice(table_ids)

        ## remove objects on table
        id2node = {node['id']: node for node in graph['nodes']}
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if
                         (edge['to_id'] == table_id) and (edge['relation_type'] == 'ON')]  # and \
        # id2node[edge['from_id']]['class_name'] not in ['rug']]
        graph = init_goal_manager.remove_obj(graph, objs_on_table)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(table_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(table_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, v, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                       only_position=table_id,
                                                       goal_obj=True)  ## add the first v objects on this table
            if not success:
                return None, None, False
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj - v, init_goal_manager.object_id_count,
                                                       objs_in_room=objs_in_room,
                                                       except_position=except_position_ids)  ## add the rest objects on other places
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'clean_table': []}
        for k, v in init_goal_manager.goal.items():
            env_goal['clean_table'].append({'take_{}_off_{}'.format(k, table_id): v})
        return graph, env_goal, True


    @staticmethod
    def put_dishwasher(init_goal_manager, graph, start=True):
        ## setup dishwasher
        # max_num_dishwasher = 4
        # num_dishwasher = init_goal_manager.rand.randint(1, max_num_dishwasher)

        # dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        # init_goal_manager.remove_obj(graph, dishwasher_ids)
        # dishwasher_position_pool = init_goal_manager.obj_position['dishwasher']
        # init_goal_manager.add_obj(graph, 'dishwasher', num_dishwasher, dishwasher_position_pool)

        dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        dishwasher_id = init_goal_manager.rand.choice(dishwasher_ids)

        ## remove objects in dishwasher
        objs_in_dishwasher = [edge['from_id'] for edge in graph['edges'] if
                              (edge['to_id'] == dishwasher_id) and (edge['relation_type'] == 'INSIDE')]
        graph = init_goal_manager.remove_obj(graph, objs_in_dishwasher)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(dishwasher_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(dishwasher_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                       objs_in_room=objs_in_room, except_position=except_position_ids,
                                                       goal_obj=True)
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'put_dishwasher': []}
        for k, v in init_goal_manager.goal.items():
            env_goal['put_dishwasher'].append({'put_{}_inside_{}'.format(k, dishwasher_id): v})
        return graph, env_goal, True


    @staticmethod
    def unload_dishwasher(init_goal_manager, graph, start=True):
        ## setup dishwasher
        # max_num_dishwasher = 4
        # num_dishwasher = init_goal_manager.rand.randint(1, max_num_dishwasher)

        # dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        # init_goal_manager.remove_obj(graph, dishwasher_ids)
        # dishwasher_position_pool = init_goal_manager.obj_position['dishwasher']
        # init_goal_manager.add_obj(graph, 'dishwasher', num_dishwasher, dishwasher_position_pool)

        dishwasher_ids = [node['id'] for node in graph['nodes'] if 'dishwasher' in node['class_name']]
        dishwasher_id = init_goal_manager.rand.choice(dishwasher_ids)

        ## remove objects in dishwasher
        objs_in_dishwasher = [edge['from_id'] for edge in graph['edges'] if
                              (edge['to_id'] == dishwasher_id) and (edge['relation_type'] == 'INSIDE')]
        graph = init_goal_manager.remove_obj(graph, objs_in_dishwasher)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(dishwasher_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(dishwasher_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, v, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                only_position=dishwasher_id,
                                                                goal_obj=True)  ## add the first v objects on this table
            if not success:
                return None, None, False
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj - v, init_goal_manager.object_id_count,
                                                                objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)  ## add the rest objects on other places
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'unload_dishwasher': []}
        for k, v in init_goal_manager.goal.items():
            env_goal['unload_dishwasher'].append({'take_{}_from_{}'.format(k, dishwasher_id): v})
        return graph, env_goal, True


    @staticmethod
    def put_fridge(init_goal_manager, graph, start=True):
        ## setup fridge
        # max_num_fridge = 4
        # num_fridge = init_goal_manager.rand.randint(1, max_num_fridge)

        # fridge_ids = [node['id'] for node in graph['nodes'] if 'fridge' in node['class_name']]
        # init_goal_manager.remove_obj(graph, fridge_ids)
        # fridge_position_pool = init_goal_manager.obj_position['fridge']
        # init_goal_manager.add_obj(graph, 'fridge', num_fridge, fridge_position_pool)

        fridge_ids = [node['id'] for node in graph['nodes'] if 'fridge' in node['class_name']]
        fridge_id = init_goal_manager.rand.choice(fridge_ids)

        ## remove objects in fridge
        objs_in_fridge = [edge['from_id'] for edge in graph['edges'] if
                          (edge['to_id'] == fridge_id) and (edge['relation_type'] == 'INSIDE')]
        graph = init_goal_manager.remove_obj(graph, objs_in_fridge)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(fridge_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(fridge_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                       objs_in_room=objs_in_room, except_position=except_position_ids,
                                                       goal_obj=True)
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'put_fridge': []}
        for k, v in init_goal_manager.goal.items():
            env_goal['put_fridge'].append({'put_{}_inside_{}'.format(k, fridge_id): v})
        return graph, env_goal, True


    @staticmethod
    def prepare_food(init_goal_manager, graph, start=True):
        # max_num_table = 4
        # num_table = init_goal_manager.rand.randint(1, max_num_table)

        # table_ids = [node['id'] for node in graph['nodes'] if 'table' in node['class_name']]
        # init_goal_manager.remove_obj(graph, table_ids)
        # table_position_pool = init_goal_manager.obj_position['table']
        # init_goal_manager.add_obj(graph, 'table', num_table, table_position_pool)

        # table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_ids = [node['id'] for node in graph['nodes'] if ('kitchentable' in node['class_name'])]
        table_id = init_goal_manager.rand.choice(table_ids)

        ## remove objects on table
        id2node = {node['id']: node for node in graph['nodes']}
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if
                         (edge['to_id'] == table_id) and (edge['relation_type'] == 'ON') and \
                         id2node[edge['from_id']]['class_name'] in ['plate', 'cutleryfork', 'waterglass', 'wineglass',
                                                                    'book', 'poundcake', 'cutleryknife']]

        graph = init_goal_manager.remove_obj(graph, objs_on_table)
        # objs_on_table = [edge['from_id'] for edge in graph['edges'] if (edge['to_id']==table_id) and (edge['relation_type']=='ON')]
        # graph = init_goal_manager.remove_obj(graph, objs_on_table)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(table_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(table_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                       objs_in_room=objs_in_room, except_position=except_position_ids,
                                                       goal_obj=True)
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'prepare_food': []}
        for k, v in init_goal_manager.goal.items():
            env_goal['prepare_food'].append({'put_{}_on_{}'.format(k, table_id): v})
        return graph, env_goal, True


    @staticmethod
    def read_book(init_goal_manager, graph, start=True):
        id2node = {node['id']: node for node in graph['nodes']}
        # table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name'])]
        for table_id in table_ids:
            for edge in graph['edges']:
                if edge['from_id'] == table_id and id2node[edge['to_id']]['class_name'] == 'livingroom':
                    break

        sofa_ids = [node['id'] for node in graph['nodes'] if ('sofa' in node['class_name'])]
        for sofa_id in sofa_ids:
            for edge in graph['edges']:
                if edge['from_id'] == sofa_id and id2node[edge['to_id']]['class_name'] == 'livingroom':
                    break

        ## remove objects on table
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if
                         (edge['to_id'] == table_id) and (edge['relation_type'] == 'ON')]  # and \
        # id2node[edge['from_id']]['class_name'] in ['plate', 'cutleryfork', 'waterglass', 'wineglass', 'book', 'poundcake']]
        graph = init_goal_manager.remove_obj(graph, objs_on_table)
        objs_on_sofa = [edge['from_id'] for edge in graph['edges'] if
                        (edge['to_id'] == sofa_id) and (edge['relation_type'] == 'ON')]  # and \
        # id2node[edge['from_id']]['class_name'] in ['plate', 'cutleryfork', 'waterglass', 'wineglass', 'book', 'poundcake']]
        graph = init_goal_manager.remove_obj(graph, objs_on_sofa)
        # objs_on_table = [edge['from_id'] for edge in graph['edges'] if (edge['to_id']==table_id) and (edge['relation_type']=='ON')]
        # graph = init_goal_manager.remove_obj(graph, objs_on_table)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(table_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(table_id)


        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k == node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                       objs_in_room=objs_in_room, except_position=except_position_ids,
                                                       goal_obj=True)
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'read_book': []}
        for k, v in init_goal_manager.goal.items():
            if k == 'book': continue
            env_goal['read_book'].append({'put_{}_on_{}'.format(k, table_id): v})
        env_goal['read_book'].append({'holds_book': 1})
        env_goal['read_book'].append({'sit_{}'.format(sofa_id): 1})
        return graph, env_goal, True

        # max_num_objs = init_goal_manager.init_pool['book']['env_max_num']
        # num_obj = init_goal_manager.rand.randint(init_goal_manager.goal['book'], max_num_objs+1)

        # target_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]
        # graph = init_goal_manager.remove_obj(graph, target_ids)
        # init_goal_manager.object_id_count, graph = init_goal_manager.add_obj(graph, 'book', num_obj, init_goal_manager.object_id_count, objs_in_room=objs_in_room, goal_obj=True)

        # except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]

        # target_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]

        # if len(target_ids)!=0:
        #     target_id = init_goal_manager.rand.choice(target_ids)

        #     if init_goal_manager.same_room:
        #         objs_in_room = init_goal_manager.get_obj_room(target_id)
        #     else:
        #         objs_in_room = None

        #     if start:
        #         init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room, except_position=except_position_ids)

        #     ## get goal
        #     env_goal = {'read_book': [{'read_{}'.format(target_id)}]}
        # else:
        #     env_goal = None
        #     # print(init_goal_manager.add_goal_obj_success)

        # return graph, env_goal


    @staticmethod
    def watch_tv(init_goal_manager, graph, start=True):
        ## add remotecontrol
        id2node = {node['id']: node for node in graph['nodes']}
        # table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name']) or ('kitchentable' in node['class_name'])]
        table_ids = [node['id'] for node in graph['nodes'] if ('coffeetable' in node['class_name'])]
        for table_id in table_ids:
            for edge in graph['edges']:
                if edge['from_id'] == table_id and id2node[edge['to_id']]['class_name'] == 'livingroom':
                    break

        sofa_ids = [node['id'] for node in graph['nodes'] if ('sofa' in node['class_name'])]
        for sofa_id in sofa_ids:
            for edge in graph['edges']:
                if edge['from_id'] == sofa_id and id2node[edge['to_id']]['class_name'] == 'livingroom':
                    break

        tv_ids = [node['id'] for node in graph['nodes'] if ('tv' in node['class_name'])]
        for tv_id in tv_ids:
            for edge in graph['edges']:
                if edge['from_id'] == tv_id and id2node[edge['to_id']]['class_name'] == 'livingroom':
                    break

        ## remove objects on table
        objs_on_table = [edge['from_id'] for edge in graph['edges'] if
                         (edge['to_id'] == table_id) and (edge['relation_type'] == 'ON')]  # and \
        # id2node[edge['from_id']]['class_name'] in ['plate', 'cutleryfork', 'waterglass', 'wineglass', 'book', 'poundcake']]
        graph = init_goal_manager.remove_obj(graph, objs_on_table)
        objs_on_sofa = [edge['from_id'] for edge in graph['edges'] if
                        (edge['to_id'] == sofa_id) and (edge['relation_type'] == 'ON')]  # and \
        # id2node[edge['from_id']]['class_name'] in ['plate', 'cutleryfork', 'waterglass', 'wineglass', 'book', 'poundcake']]
        graph = init_goal_manager.remove_obj(graph, objs_on_sofa)
        # objs_on_table = [edge['from_id'] for edge in graph['edges'] if (edge['to_id']==table_id) and (edge['relation_type']=='ON')]
        # graph = init_goal_manager.remove_obj(graph, objs_on_table)

        if init_goal_manager.same_room:
            objs_in_room = init_goal_manager.get_obj_room(table_id)
        else:
            objs_in_room = None

        except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        except_position_ids.append(table_id)

        for k, v in init_goal_manager.goal.items():
            obj_ids = [node['id'] for node in graph['nodes'] if k in node['class_name']]
            graph = init_goal_manager.remove_obj(graph, obj_ids)

            num_obj = init_goal_manager.rand.randint(v, init_goal_manager.init_pool[k]['env_max_num'] + 1)  # random select objects >= goal
            init_goal_manager.object_id_count, graph, success = init_goal_manager.add_obj(graph, k, num_obj, init_goal_manager.object_id_count,
                                                       objs_in_room=objs_in_room, except_position=except_position_ids,
                                                       goal_obj=True)
            if not success:
                return None, None, False

        if start:
            init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room,
                                                                except_position=except_position_ids)

        ## get goal
        env_goal = {'watch_tv': []}
        for k, v in init_goal_manager.goal.items():
            if k in ['tv', 'remotecontrol']: continue
            env_goal['watch_tv'].append({'put_{}_on_{}'.format(k, table_id): v})
        env_goal['watch_tv'].append({'turnOn_{}'.format(tv_id): 1})
        env_goal['watch_tv'].append({'holds_remotecontrol': 1})
        env_goal['watch_tv'].append({'sit_{}'.format(sofa_id): 1})

        return graph, env_goal, True

        # max_num_objs = init_goal_manager.init_pool['remotecontrol']['env_max_num']
        # num_obj = init_goal_manager.rand.randint(init_goal_manager.goal['remotecontrol'], max_num_objs+1)

        # target_ids = [node['id'] for node in graph['nodes'] if 'remotecontrol' in node['class_name']]
        # if len(target_ids)==0:
        #     init_goal_manager.object_id_count, graph = init_goal_manager.add_obj(graph, 'remotecontrol', num_obj, init_goal_manager.object_id_count, objs_in_room=objs_in_room, goal_obj=True)
        #     target_ids = [node['id'] for node in graph['nodes'] if 'book' in node['class_name']]

        # assert len(target_ids)!=0
        # target_id = init_goal_manager.rand.choice(target_ids)

        # if init_goal_manager.same_room:
        #     objs_in_room = init_goal_manager.get_obj_room(target_id)
        # else:
        #     objs_in_room = None

        # ## set TV off
        # tv_ids = [node['id'] for node in graph['nodes'] if 'tv' in node['class_name']]
        # tv_id = init_goal_manager.rand.choice(tv_ids)
        # graph = init_goal_manager.set_tv_off(graph, tv_id)

        # ## set other objects
        # except_position_ids = [node['id'] for node in graph['nodes'] if ('floor' in node['class_name'])]
        # if start:
        #     init_goal_manager.object_id_count, graph = init_goal_manager.setup_other_objs(graph, init_goal_manager.object_id_count, objs_in_room=objs_in_room, except_position=except_position_ids)

        # ## get goal
        # env_goal = {'watch_tv': [ {'on_{}'.format(tv_id)}, {'grab_{}'.format(target_id)} ]}
        # return graph, env_goal


    @staticmethod
    def setup_table_prepare_food(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['setup_table'].keys()}
        graph, env_goal1, success = Task.setup_table(init_goal_manager, graph)
        # pdb.set_trace()
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['prepare_food'].keys()}

        graph, env_goal2, success = Task.prepare_food(init_goal_manager, graph, start=False)
        init_goal_manager.goal = all_goal
        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        # pdb.set_trace()
        return graph, env_goal1, success


    @staticmethod
    def setup_table_read_book(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['setup_table'].keys()}
        graph, env_goal1, success = Task.setup_table(init_goal_manager, graph)
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['read_book'].keys()}
        graph, env_goal2, success = Task.read_book(init_goal_manager, graph, start=False)

        init_goal_manager.goal = all_goal
        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success


    @staticmethod
    def setup_table_watch_tv(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['setup_table'].keys()}
        graph, env_goal1, success = Task.setup_table(init_goal_manager, graph)
        if not success:
            return None, None, False
        graph, env_goal2, success = Task.watch_tv(init_goal_manager, graph, start=False)
        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success


    @staticmethod
    def setup_table_put_fridge(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['setup_table'].keys()}
        graph, env_goal1, success = Task.setup_table(init_goal_manager, graph)
        # pdb.set_trace()
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['put_fridge'].keys()}
        graph, env_goal2, success = Task.put_fridge(init_goal_manager, graph, start=False)
        init_goal_manager.goal = all_goal
        if not success:
            # pdb.set_trace()
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success


    @staticmethod
    def setup_table_put_dishwasher(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['setup_table'].keys()}
        graph, env_goal1, success = Task.setup_table(init_goal_manager, graph)
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['put_dishwasher'].keys()}
        graph, env_goal2, success = Task.put_dishwasher(init_goal_manager, graph, start=False)
        init_goal_manager.goal = all_goal

        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success

    @staticmethod
    def prepare_food_put_dishwasher(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['prepare_food'].keys()}
        graph, env_goal1, success = Task.prepare_food(init_goal_manager, graph)
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['put_dishwasher'].keys()}
        graph, env_goal2, success = Task.put_dishwasher(init_goal_manager, graph, start=False)
        init_goal_manager.goal = all_goal

        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success

    @staticmethod
    def put_fridge_put_dishwasher(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['put_fridge'].keys()}
        graph, env_goal1, success = Task.put_fridge(init_goal_manager, graph)
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['put_dishwasher'].keys()}
        graph, env_goal2, success = Task.put_dishwasher(init_goal_manager, graph, start=False)
        init_goal_manager.goal = all_goal

        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success

    @staticmethod
    def put_dishwasher_read_book(init_goal_manager, graph):
        all_goal = init_goal_manager.goal
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['put_dishwasher'].keys()}
        graph, env_goal1, success = Task.put_dishwasher(init_goal_manager, graph)
        if not success:
            return None, None, False
        init_goal_manager.goal = {obj_name: value for obj_name, value in all_goal.items() if
                                  obj_name in init_goal_manager.init_pool_tasks['read_book'].keys()}
        graph, env_goal2, success = Task.read_book(init_goal_manager, graph, start=False)
        init_goal_manager.goal = all_goal

        if not success:
            return None, None, False
        env_goal1.update(env_goal2)
        return graph, env_goal1, success
