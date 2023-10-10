# TDW Multi-Agent Transport

## Codebase Layouts 

```
|__ tdw_gym/ 					main code
|       |__ challenge.py         main evaluation code
|       |__ tdw_gym.py           main env code
|       |__ h_agent.py           RHP Agent
|       |__ lm_agent.py          CoELA
|
|__ scene_generator/ 			code for generating dataset
|
|__ dataset/ 				dataset configuration & dataset storage
|
|__ transport_challenge_multi_agent/    low level env controller
|
|__ scripts/ 				scripts for running experiments
|
|__ LLM                         CoELA
```

## Setup

Run the following commands step by step to setup the default environments:

```bash
conda create -n tdw_mat python=3.9
conda activate tdw_mat
pip3 install -e .
```

If you're running TDW on a remote Linux server, follow the [TDW Installation Document](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md) to configure the X server.

After that, you can run the demo scene to verify your setup:

```bash
python demo/demo_scene.py
```

*For vision detection module:* If you want to install the vision detection module, please install `mmdetection`:
```bash
pip install torch torchvision torchaudio
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```
If you encounter trouble when installing `mmdetection`, please refer to [here](https://mmdetection.readthedocs.io/en/latest/get_started.html) for detailed install guidance.

## Run Experiments

We prepare the example scripts to run experiments with HP baseline and our Cooperative LLM Agent under the folder `scripts` and `scripts/wo_gt_mask`. For example, to run experiments with two LLM Agents, run the following command:

```bash
./scripts/test_LMs.sh
```

Download `transport challenge asset bundles`: Commonly it is automatically downloaded when running the scripts. If you encounter problems, you can download it [here](https://drive.google.com/file/d/1us2hpJj3_u1Ti_R0OrqVDgUQbdMPUaKN/view?usp=sharing), and unzip it in the `TDW_MAT` folder.

## Detection Model

Besides using ground truth segmentation mask in `TDW_MAT`, we also have `no-gt-mask` mode. Here you need to train a segmentation model on your own.

We finetune Mask R-CNN as our detection baseline, which is based on `mmdetection`. You can download the model weight [here](https://drive.google.com/file/d/1JTrV5jdF-LQVwY3OsV3Jd3r6PRghyHBp/view?usp=sharing). If you want to use it, put it in the `detection_pipeline/` folder (where you can also find the `config` file). 

To test the installation of our detection model and the pre-trained model, you can run:

```bash
python detection_pipeline/test_install.py
```

By adding `--no_gt_mask` in the scripts, the env will not provide ground truth segmentation masks anymore, and thus the agents need to detect them. 

## Environment Details

We extend the [ThreeDWorld Transport Challenge](https://arxiv.org/abs/2103.14025) into a multi-agent setting with more types of objects and containers, more realistic object placements, and support communication between agents, named ThreeDWorld Multi-Agent Transport (TDW-MAT), built on top of the [TDW platform](https://www.threedworld.org/). 

The agents are tasked to transport as many target objects as possible to the goal position with the help of containers as tools. One container can carry most three objects, and without containers, the agent can transport only two objects at a time. The agents have the ego-centric visual observation and action space as before with a new communication action added.

### Tasks 

We selected $6$ scenes from the TDW-House dataset and sampled $2$ types of tasks and $2$ settings in each of the scenes, making a test set of $24$ episodes. Every scene has $6$ to $8$ rooms, $10$ objects, and a few containers. An episode is terminated if all the target objects have been transported to the goal position or the maximum number of frames ($3000$) is reached. 

The tasks are named `food task` and `stuff task`. Containers for the `food task` can be found in both the kitchen and living room, while containers for the `stuff task` can be found in the living room and office. 

The configuration and distribution of containers vary based on two distinct settings: the `Enough Container Setting` and the `Rare Container Setting`. In the `Enough Container Setting`, the ratio of containers to objects stands at $1:2$, and containers associated with a specific task are located in no more than two rooms. On the other hand, in the `Rare Container Setting`, the container-to-object ratio decreases to $1:5$. This distribution differs from the "Enough Container Setting" as containers in the `Rare Container Setting` are strictly localized to a single room. 

One example of scenes, target objects, and containers is shown in the following image:

![task_description_tdw](../../assets/task_description_tdw.png)

### Metrics

  - **Transport Rate (TR)**: The fraction of the target objects successfully transported to the goal position.
  - **Efficiency Improvements (EI)**: The efficiency improvements of cooperating with base agents.

### Multi-agent Asynchronized Setting

Agents may take different numbers of frames to finish (or fail) one action, one env step is finished until any agent's action is not ongoing, and the current obs is returned to all agents.
All agents are asked for a new action, and agents having ongoing actions will directly switch to the new action if actions change. 

### Gym Scenes

The dataset is modular in its design, consisting of several physical floor plan geometries with wall and floor texture 
variations (e.g. parquet flooring, ceramic tile, stucco, carpet, etc.) and various furniture and prop layouts (tables, 
chairs, cabinets, etc.), for a total of 6 separate environments. Every scene has 6 to 8 rooms, 10 objects, and 4 containers.

### Gym Observations
```
"rgb": Box(0, 512, (3, 512, 512))
"depth": Box(0, 512, (512, 512))
"seg_mask": (0, 512, (512, 512, 3))
"agent": Box(-30, 30, (6,)) # position
"status": ActionStatus[ongoing, success,fail,...]
"camera_matrix": Box(-30, 30, (4, 4)),
"valid": bool[True / False] # Whether the last action is valid
"held_objects": [{'id': id/id/None, 'type': 0/1/None, 'name': name/name/None',  contained': [None] * 3 / [..] / [None] * 3}, {'id': id/id/None, 'type': 0/1/None, 'name': name/name/None', 'contained': [None] * 3 / [..] / [None] * 3}],
"oppo_held_objects": [{'id': id/id/None, 'type': 0/1/None, 'name': name/name/None', contained': [None] * 3 / [..] / [None] * 3}, {'id': id/id/None, 'type': 0/1/None, 'name': name/name/None', 'contained': [None] * 3 / [..] / [None] * 3}], # we can see it when opponent is visible
"current_frames": int
'visible_objects': [{'id': ..., 'type': ..., 'seg_color': ..., 'name': ...}]
"containment": {"agent_id": {"visible": True / False, "contained": [id, ...]}}
"messages": (Text(max_length=1000, charset=string.printable), Text(max_length=1000, charset=string.printable))
```

### Api the agent can access
```
belongs_to_which_room((x, y, z)) -> room_name: given a location, return the room name;
center_of_room(room_name) -> [x, y, z]: given a room name, return the center of the room;
check_pos_in_room(x, z) -> bool: given a location, return whether it is inside the floorplan.
```

### Gym resetting
reset gym with {scene, layout, task}, and will return:
```
"goal_description": a dict {'target_name_i': num_i}
"room_type": a list [room_name_0, ...]
```

### Gym Actions
* move forward at 0.5
```
dict {"type": 0} 
```
* turn left 15 degrees
```
dict {"type": 1} 
```
* turn right 15 degrees
```
dict {"type": 2} 
```
* grasp the object with the arm
```
dict {"type": 3, "object": object_id, "arm": "left" or "right"} 
```
* put the holding object into the holding container
```
dict {"type": 4} 
```
* drop objects
```
dict {"type": 5, "arm": "left" or "right"}
```
* send messages
```
dict {"type": 6, "message": Text}
```
