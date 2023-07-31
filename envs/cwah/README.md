# Communicative Watch-And-Help

## Setup
### Step 1: Get the VirtualHome Simulator and API
Clone the [VirtualHome API](https://github.com/xavierpuigf/virtualhome.git) repository one folder above this repository

```bash
cd ..
git clone --branch wah https://github.com/xavierpuigf/virtualhome.git
```

Download the simulator, and put it in an `executable` folder, one folder above this repository


- [Download](https://drive.google.com/file/d/1JTrV5jdF-LQVwY3OsV3Jd3r6PRghyHBp/view?usp=sharing) Linux x86-64 version.


**[IMPORTANT]** Please use our modified version of the VirtualHome repo (wah branch in the VirtualHome repo) and the v2.3.0 version of the executable.

The files should be organized as follows:

```
|
|--cwah/
|
|--virtualhome/
|
|--executable/
```

### Step 2: Install Requirements

`cd cwah`

`conda create --name cwah python=3.8`

[//]: # (&#40;or use `conda create --prefix /work/pi_name/$USER-conda/envs/cwah python=3.8` if you want to create in a specific folder&#41;)

`conda activate cwah`

`pip install -r requirements.txt `

## Run Experiments

We prepare the example scripts to run experiments with HP baseline and our Cooperative LLM Agent under the folder `scripts`. For example, to run experiments with two LLM Agents, run the following command:

```bash
./scripts/symbolic_obs_llm_llm.sh
```

For more details on the arguments, please refer to the scripts and `arguments.py`.

## User Interface

To view the interface for human experiments, run the following command, and access the interface at `localhost:5005`.

```bash
cd virtualhome_userinterface
python vh_demo.py --deployment remote --executable_file ../../executable/linux_exec.v2.3.0.x86_64 --extra_agent MCTS_comm --task_group 0 --showmodal
```

<!-- --task_group is recommended to be chosen in 0, 5, 10, 16, 20, 26, 30, 32, 40, 49. **Only one number should be chosen though --task_group is nargs**.

--extra_agent choices = ["MCTS_comm", "LLM_comm", "LLM", "none"], default='none', meaning working with MCTS agent, working with LLM agent that can communicate, working with LLM agent that cannot communicate, or working alone.

--showmodal is used to let the questionnaire appear after you have completed the task. -->
