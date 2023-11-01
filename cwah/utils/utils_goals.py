def convert_goal_spec(task_name, goal, state, exclude=[]):
    """
    Convert the task goal into a format interpreted by the planner and model
    """
    goals = {}
    containers = [[node['id'], node['class_name']] for node in state['nodes'] if
                  node['class_name'] in ['kitchencabinets', 'kitchencounterdrawer', 'kitchencounter']]
    id2node = {node['id']: node for node in state['nodes']}
    for key_count in goal:
        key = list(key_count.keys())[0]
        count = key_count[key]
        elements = key.split('_')
        print(elements)
        if elements[1] in exclude: continue
        if task_name in ['setup_table', 'prepare_food']:
            predicate = 'on_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name in ['put_dishwasher', 'put_fridge']:
            predicate = 'inside_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name == 'clean_table':
            predicate = 'offOn'
            predicate = 'offOn_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name == 'unload_dishwahser':
            predicate = 'offInside'
            predicate = 'offOn_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        elif task_name == 'read_book':
            if elements[0] == 'holds':
                predicate = 'holds_{}_{}'.format('book', 1)
            elif elements[0] == 'sit':
                predicate = 'sit_{}_{}'.format(1, elements[1])
            else:
                predicate = 'on_{}_{}'.format(elements[1], elements[3])
                # count = 0
            goals[predicate] = count
        elif task_name == 'watch_tv':
            if elements[0] == 'holds':
                predicate = 'holds_{}_{}'.format('remotecontrol', 1)
            elif elements[0] == 'turnOn':
                predicate = 'turnOn_{}_{}'.format(elements[1], 1)
            elif elements[0] == 'sit':
                predicate = 'sit_{}_{}'.format(1, elements[1])
            else:
                predicate = 'on_{}_{}'.format(elements[1], elements[3])
            goals[predicate] = count
        else:
            predicate = key
            goals[predicate] = count

    return goals
