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


**[IMPORTANT]** Please use our modified version of the VirtualHome API repo (wah branch in the VirtualHome repo) and the provided version of the executable.

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

## Environment Details

Communicative Watch-And-Help(C-WAH) is an extension of the [Watch-And-Help challenge](https://github.com/xavierpuigf/watch_and_help), which enables agents to send messages to each other. Sending messages, alongside other actions, takes one timestep and has an upper limit on message length.

### Tasks 

Five types of tasks are available in C-WAH, named `Prepare afternoon tea`, `Wash dishes`, `Prepare a meal`, `Put groceries`, and `Set up a dinner table`. These tasks include a range of housework, and each task contains a few subgoals, which are described by predicates. A predicate is in `ON/IN(x, y)` format, that is, `Put x ON/IN y`. The detailed descriptions of tasks are listed in the following table:

| Task Name | Predicate Set |
| ------- | ------- |
| Prepare afternoon tea   | ON(cupcake,coffeetable), ON(pudding,coffeetable), ON(apple,coffeetable), ON(juice,coffeetable), ON(wine,coffeetable)  |
| Wash dishes  | IN(plate,dishwasher), IN(fork,dishwasher)  |
| Prepare a meal | ON(coffeepot,dinnertable),ON(cupcake,dinnertable), ON(pancake,dinnertable), ON(poundcake,dinnertable), ON(pudding,dinnertable), ON(apple,dinnertable), ON(juice,dinnertable), ON(wine,dinnertable) |
|Put groceries | IN(cupcake,fridge), IN(pancake,fridge), IN(poundcake,fridge), IN(pudding,fridge), IN(apple,fridge), IN(juice,fridge), IN(wine,fridge) |
|Set up a dinner table | ON(plate,dinnertable), ON(fork,dinnertable) |

The task goal is to satisfy all the given subgoals within $250$ time steps, and the number of subgoals in each task ranges from $3$ to $5$. 

### Metrics

  - **Average Steps (L)**: Number of steps to finish the task;
  - **Efficiency Improvement (EI)**: The efficiency improvements of cooperating with base agents.

## User Interface

To view the interface for human experiments, run the following command, and access the interface at `localhost:5005`.

```bash
cd virtualhome_userinterface
python vh_demo.py --deployment remote --executable_file ../../executable/linux_exec.v2.3.0.x86_64 --extra_agent MCTS_comm --task_group 0 --showmodal
```

<!-- --task_group is recommended to be chosen in 0, 5, 10, 16, 20, 26, 30, 32, 40, 49. **Only one number should be chosen though --task_group is nargs**.

--extra_agent choices = ["MCTS_comm", "LLM_comm", "LLM", "none"], default='none', meaning working with MCTS agent, working with LLM agent that can communicate, working with LLM agent that cannot communicate, or working alone.

--showmodal is used to let the questionnaire appear after you have completed the task. -->
