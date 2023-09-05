# TDW Multi-Agent Transport

## Codebase Layouts 

```
|__ tdw-gym/ 					main code
|       |__ challenge.py         main evaluation code
|       |__ tdw_gym.py           main env code
|       |__ h_agent.py           HP Agent
|       |__ lm_agent.py          Cooperative LLM Agent
|
|__ scene_generator/ 			code for generating dataset
|
|__ dataset/ 					dataset configuration & dataset storage
|
|__ transport_challenge_multi_agent/ low level controller
|
|__ scripts/ 					scripts for running experiments
```

## Setup

Run the following commands step by step to setup the default environments:

*For vision detection module:* If you want to install the vision detection module, replace `requirements.txt` with `requirements_with_vision.txt` before running the following command.

```bash
cd tdw_mat
conda create -n tdw_mat python=3.9
conda activate tdw_mat
pip3 install -e .
```

If you're running TDW on a remote Linux server, follow the [TDW Installation Document](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md) to configure the X server.

After that, you can run the demo scene to verify your setup:

```bash
python demo/demo_scene.py
```

## Run Experiments

We prepare the example scripts to run experiments with HP baseline and our Cooperative LLM Agent under the folder `scripts` and `scripts/wo_gt_mask`. For example, to run experiments with two LLM Agents, run the following command:

```bash
./scripts/test_LMs.sh
```

Download `transport challenge asset bundles`: Commonly it is automatically downloaded when running the scripts. If you meet problem, you can download it [here](https://drive.google.com/file/d/1us2hpJj3_u1Ti_R0OrqVDgUQbdMPUaKN/view?usp=sharing), and unzip it in the `TDW_MAT` folder.

## Detection Model

Besides use ground truth segmentation mask in `TDW_MAT`, we also have `no-gt-mask` mode. Here you need to train a segmentation model by your own.

We finetune a Resnet model as our detection baseline, which is based on `mmdetection`. You can download the model weight [here](https://drive.google.com/file/d/1JTrV5jdF-LQVwY3OsV3Jd3r6PRghyHBp/view?usp=sharing). If you want to use it, put it in `detection_pipeline/` folder (where you can also find the `config` file). 

To test the installation of our detection model, you can run:

```bash
python detection_pipeline/test_install.py
```

By add `--no_gt_mask` in the scripts, the env will not provide ground truth segmentation mask anymore, and thus the agents need to detect them. 

## More details on the environment

### Multi-agent Asynchronized Setting

Agents may take different number of frames to finish (or fail) one action, one env step is finished until any agent's action is not ongoing, and the current obs is returned to all agents.
All agents are asked for a new action, and agents having ongoing actions will directly switch to the new action if actions changed. 

### Gym Scenes

The dataset is modular in its design, consisting of several physical floor plan geometries with a wall and floor texture 
variations (e.g. parquet flooring, ceramic tile, stucco, carpet etc.) and various furniture and prop layouts (tables, 
chairs, cabinets etc.), for a total of 6 separate environments. Every scene has 6 to 8 rooms, 10 objects, and 4 containers.

### Gym Observations
```
"rgb": Box(0, 256, (3, 256, 256))
"depth": Box(0, 256, (256, 256))
"seg_mask": (0, 256, (256, 256, 3))
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
* grasp the object with arm
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
