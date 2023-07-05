import os
import json
import numpy as np
import sys

settings = sys.argv[1:]
dataset_prefix = "dataset/arxiv_dataset_v6/"
if len(settings) > 0: dataset_prefix = f"dataset/{settings[0]}/"
file_list = os.listdir(dataset_prefix)
room_list = {"Livingroom" : 0, "Office" : 1, "Kitchen" : 2}
room_name = ["Livingroom", "Office", "Kitchen"]
each_room_count = np.array([0, 0, 0])
object_name = ["apple", "banana", "orange", "bread", "loaf_bread", "burger", "pen", "calculator", "mouse", "purse", "iphone", "lighter", "bowl", "plate", "tea_tray", "plastic_basket", "wicker_basket", "wood_basket"]
object_list = {"apple": 0, "banana": 1, "orange": 2, "bread": 3, "loaf_bread": 4, "burger": 5, "pen": 6, "calculator": 7, "mouse": 8, "purse": 9, "iphone": 10, "lighter": 11, "bowl": 12, "plate": 13, "tea_tray": 14, "plastic_basket": 15, "wicker_basket": 16, "wood_basket": 17}
object_count = 18

count = [[0 for i in range(3)] for j in range(100)]
with open("dataset/name_map.json", 'r') as f:
    name_map = json.load(f)
with open('dataset/room_types.json', 'r') as f:
    room_functionals = json.loads(f.read())

for file in file_list:
    if 'metadata' not in file: continue
    scene_id = file[0]
    layout_id = int(file[3])
    for room in room_functionals[scene_id][layout_id]:
        if room in room_list.keys():
            each_room_count[room_list[room]] += 1
    with open(dataset_prefix + file, 'r') as f:
        json_data = json.load(f)
    object_room_list = json_data['object_room_list']
    for object_room in object_room_list:
        if name_map[object_room[0]] not in object_list.keys():
            object_list[name_map[object_room[0]]] = object_count
            object_count += 1
            object_name.append(name_map[object_room[0]])
        count[object_list[name_map[object_room[0]]]][room_list[object_room[1]]] += 1
count = count[:object_count]

for i in range(object_count):
    for j in range(3):
        count[i][j] /= each_room_count[j]

object_sum = np.sum(count, axis = 1)

for i in range(object_count):
    count[i] = np.array([count[i][j] / object_sum[i] for j in range(3)])
    print(object_name[i], count[i])

import numpy as np
import matplotlib.pyplot as plt

density = np.array(count).transpose()

plt.imshow(density, interpolation='nearest', cmap = 'YlGnBu')
plt.xticks(np.arange(len(object_name)), object_name, rotation=90)
plt.yticks(np.arange(len(room_name)), room_name, rotation=0)
for i in range(density.shape[0]):
    for j in range(density.shape[1]):
        plt.text(j, i, round(density[i, j], 2), ha='center', va='center',
                 color='black' if density[i, j] < 0.5 else 'white', fontsize=6)
plt.colorbar()
plt.tight_layout()

plt.savefig(f"{dataset_prefix}object_density.png")