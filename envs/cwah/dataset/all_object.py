import pickle
all_objects = set()
env_task_set = pickle.load(open('./dataset/test_env_set_help_filtered.pik', 'rb'))
for i in range(len(env_task_set)):
    for k in env_task_set[i]['task_goal'].keys():
        for t in env_task_set[i]['task_goal'][k].keys():
            objects = t.split('_')[1:]
            for o in objects:
                if (o.isdigit()):
                    o = [node['class_name'] for node in env_task_set[i]['init_graph']['nodes'] if node['id'] == int(o)][0]
                all_objects.add(o)
print(all_objects)

# 'fridge', 'plate', 'coffeetable', 'poundcake', 'wine', 'juice', 'pudding', 'apple', 'dishwasher', 'pancake', 'cutleryfork', 'kitchentable', 'cupcake', 'coffeepot'