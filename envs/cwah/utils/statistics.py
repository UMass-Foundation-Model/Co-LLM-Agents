#logstoresults.py------------------------------------------------------------
import argparse
import pickle
from pathlib import Path

import pandas as pd
import os
import sys
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import re
from copy import deepcopy

parser = argparse.ArgumentParser()
'''old dataset'''
#parser.add_argument("--dataset_path", default='/work/pi_chuangg_umass_edu/sjm_gitsync/watch_and_help/dataset/test_env_set_help_filtered.pik', type=str)
#parser.add_argument('--test_task', default=(0, 4, 22, 30, 40, 46, 60, 62, 80, 85), type=int, nargs='+',
#					help='task ids to be tested')
'''new dataset'''
parser.add_argument('--test_task', default=(0, 5, 10, 16, 20, 26, 30, 32, 40, 49), type=int, nargs='+',
					help='task ids to be tested')
parser.add_argument("--dataset_path", default='/work/pi_chuangg_umass_edu/sjm_gitsync/watch_and_help/dataset/test_env_set_help.pik', type=str)

parser.add_argument("--record_dir", type=str, nargs='+', required=True, help="the folders that you want to handle")
parser.add_argument("--method_names", type=str, nargs='+', help='this is used to plot')
parser.add_argument("--num_runs", default=5, type=int)
parser.add_argument("--num_per_task", default=2, type=int)
parser.add_argument("--generate_result", action='store_true')
parser.add_argument('--also_generate_single_dir_result', action='store_true')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--remove_comm_action", action='store_true')
parser.add_argument("--action_type", default = 'L' , choices=['L', 'S', 'B', 'E', 'None', 'others'], type=str, help='steps kinds that are used in plot')
parser.add_argument("--analyse", action='store_true')
parser.add_argument("--analyse_std", action='store_true')
parser.add_argument("--subgoal_analysis", action='store_true')
parser.add_argument("--single_dir", default='/work/pi_chuangg_umass_edu/sjm_gitsync/test_results/vision4.3/single_vision__normal_image_2', type=str) # visual
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--plot", action='store_true')
parser.add_argument('--plotpie', action='store_true', help = 'plot pie chart of 5 kinds of action, do not need to set plot=true')
parser.add_argument("--errorbar", action='store_true')
parser.add_argument("--split_show", action='store_true')
parser.add_argument('--metric', default='StepNumber', type=str, choices=['SuccessRate', 'StepNumber', 'SpeedUp', 'EI', 'Reward', 'DuplicateSubgoalRate', 'NoUseSubgoalRate', 'InfluenceRate'],
					help='task metric to be plotted')
parser.add_argument('--ylim', default='(None, None)', type=str,
					help='set ylim')
parser.add_argument('--chdir', default='.', type=str)
parser.add_argument('--results_pik_name', default='results.pik', type=str)
#plot_para
parser.add_argument('--ncol', default=2, type=int)
parser.add_argument('--left', default=None, type=int)
parser.add_argument('--right', default=None, type=int)
parser.add_argument('--bottom', default=None, type=int)
parser.add_argument('--top', default=0.8, type=int)
## log args
parser.add_argument("--print_log", action='store_true')
parser.add_argument("--agent_id", nargs='+', default=(0, 1), type=int)
parser.add_argument("--output_dir", default='Output')
parser.add_argument("--current_log", action='store_true')

args = parser.parse_args()

args.ylim = eval(args.ylim)
if args.method_names is None:
	args.method_names = [(record_dir[:-1] if record_dir[-1] == '/' else record_dir).split('/')[-1] for record_dir in args.record_dir]

metric_map = {'SuccessRate' : 'Success Rate', 'StepNumber' : 'Step Number', 'SpeedUp' : 'Speed Up', 'EI' : 'EI', 'Reward' : 'Reward', 'DuplicateSubgoalRate': 'Duplicate Subgoal Rate', 'NoUseSubgoalRate': 'No Use Subgoal Rate', 'InfluenceRate': 'Influence Rate'}
args.metric = metric_map[args.metric]
task_names = ['read_book', 'put_dishwasher', 'prepare_food', 'put_fridge', 'setup_table']

if not args.print_log:
	assert(len(list(args.record_dir)) == len(list(args.method_names)))
if args.also_generate_single_dir_result:
	assert args.generate_result

pwd = os.getcwd()
os.chdir(args.chdir)
env_task_set = pickle.load(open(args.dataset_path, 'rb'))
episode_ids = list(range(len(env_task_set)))
test_task = episode_ids
num_per_task = args.num_per_task
num_runs = args.num_runs
cache_file = os.path.join(pwd, 'Output', 'test_results.pik')
if num_per_task < 10:
	test_task = args.test_task
single_result = None

pie_chart_keys = ['comm', 'S', 'B', 'E', 'None', 'others', 'navigation', 'interaction']
pie_chart_plot_keys = ['comm', 'None', 'others', 'navigation', 'interaction']


def generate_result():
	generate_dir = list(args.record_dir)
	if args.also_generate_single_dir_result and args.single_dir not in generate_dir:
		generate_dir.append(args.single_dir)
	for record_dir in generate_dir:
		pattern = re.compile(r'logs_agent_\d+_.*_(\d+)\.pik')
		max_num_1 = -1
		for filename in os.listdir(record_dir):
			file_path = os.path.join(record_dir, filename)
			if os.path.isfile(file_path):
				match = pattern.match(filename)
				if match:
					num_1 = int(match.group(1))
					max_num_1 = max(max_num_1, num_1)

		test_results = {}
		env_id = [[] for _ in range(len(episode_ids))]
		S = [[] for _ in range(len(episode_ids))]
		L = [[] for _ in range(len(episode_ids))]
		L_remove_comm = [[] for _ in range(len(episode_ids))]
		dup_s = [[] for _ in range(len(episode_ids))]
		nouse_s = [[] for _ in range(len(episode_ids))]
		step_data = [[] for _ in range(len(episode_ids))]
		cnt_influences = [[] for _ in range(len(episode_ids))]
		results_path = os.path.join(record_dir, args.results_pik_name)
		steps_list, failed_tasks = [], []
		if os.path.isfile(results_path) and not args.overwrite:
			error_msg = f"{results_path} already exists, please use --overwrite to overwrite"
			raise ValueError(error_msg)

		# steps_list, failed_tasks = [], []
		have_nouse = True
		for seed in range(max_num_1 + 1):
			for episode_id in episode_ids:
				curr_log_file_name = os.path.join(record_dir, 'logs_agent_{}_{}_{}.pik'.format(
					env_task_set[episode_id]['task_id'],
					env_task_set[episode_id]['task_name'],
					seed))

				if not os.path.isfile(curr_log_file_name):
					continue
					# raise ValueError(f"{curr_log_file_name} does not exist")
				assert len(S[episode_id]) == seed, f"{S[episode_id]}, {seed}"
				with open(curr_log_file_name, 'rb') as fd:
					file_data = pickle.load(fd)
				S[episode_id].append(file_data['finished'])
				env_id[episode_id].append(file_data['env_id'])
				two_agent = len(file_data['action'][1]) != 0
				cnt_duplicate_subgoal = 0
				min_subgoal_len = min(len(file_data['subgoals'][0]), len(file_data['subgoals'][1])) if two_agent else len(file_data['subgoals'][0])
				if two_agent:
					for i in range(min_subgoal_len):
						if file_data['subgoals'][0][i] == file_data['subgoals'][1][i]:
							cnt_duplicate_subgoal += 1
					for i in reversed(range(min_subgoal_len)):
						if i == 0 or file_data['subgoals'][0][i] == file_data['subgoals'][1][i] == file_data['subgoals'][0][i-1] == file_data['subgoals'][1][i-1]:
							cnt_duplicate_subgoal -= 1
						else:
							break
				cnt_influence = 0
				comm_num = 0
				if two_agent:
					# calculate one agent send message and another agent change its subgoal
					# for i in range(1, min_subgoal_len):
					#     if file_data['subgoals'][1][i] != file_data['subgoals'][1][i-1]:
					#         if file_data['action'][0][i-1] is not None and file_data['action'][0][i].startswith('[send_message]'):
					#             cnt_influence += 1
					#     if file_data['subgoals'][0][i] != file_data['subgoals'][0][i-1]:
					#         if file_data['action'][1][i-1] is not None and file_data['action'][1][i].startswith('[send_message]'):
					#             cnt_influence += 1
					#another calc method
					for i in range(0, min_subgoal_len-1):
						if file_data['action'][0][i] is not None and file_data['action'][0][i].startswith('[send_message]'):
							comm_num += 1
							if file_data['subgoals'][1][i] != file_data['subgoals'][1][i+1]:
								cnt_influence += 1
						if file_data['action'][1][i] is not None and file_data['action'][1][i].startswith('[send_message]'):
							comm_num += 1
							if file_data['subgoals'][0][i] != file_data['subgoals'][0][i+1]:
								cnt_influence += 1
				cnt_influences[episode_id].append(cnt_influence/comm_num if comm_num != 0 else 0)
				assert not two_agent or len(file_data['action'][0]) == len(file_data['action'][1])
				if two_agent:
					agent = {k:[0,0] for k in pie_chart_keys}
				else:
					agent = {k:[0] for k in pie_chart_keys}
				agent_index = [0] if not two_agent else [0,1]
				for index in agent_index:
					for i in file_data['action'][index]:
						if i is None:
							agent['None'][index] += 1
						else:
							act = i[i.find('[') + 1 :i.find(']')]
							if act == 'send_message':
								if "'S':" in i:
									agent['S'][index] += 1
								if "'B':" in i:
									agent['B'][index] += 1
								if "'E':" in i:
									agent['E'][index] += 1
								agent['comm'][index] += 1
							elif act in ['walktowards', 'turnleft', 'TurnLeft']:
								agent['navigation'][index] += 1
							elif act in ['open', 'close', 'grab', 'putin', 'putback']:
								agent['interaction'][index] += 1
							else:
								print(i)
								assert False

				# agent_means = {k:v for k,v in agent_0.items()} if not two_agent else {k:(agent_0[k]+agent_1[k])/2 for k,v in agent_0.keys()}
				steps = len(file_data['action'][0])
				steps_remove_comm = max(len([i for i in file_data['action'][0] if ((not i) or (not i.startswith('[send_message]')))]), len([i for i in file_data['action'][1] if ((not i) or (not i.startswith('[send_message]')))]))
				if 'cnt_nouse_subgoal'  not in file_data.keys():
					have_nouse = False
				if have_nouse:
					cnt_nouse_subgoal = file_data['cnt_nouse_subgoal']
				# print(file_data['action'][0])
				# steps_remove_comm = agent_means['others']
				if file_data['finished']:
					steps_list.append(steps)
				else:
					failed_tasks.append(file_data['task_name'])
				L[episode_id].append(steps)
				L_remove_comm[episode_id].append(steps_remove_comm)
				step_data[episode_id].append(agent)
				dup_s[episode_id].append(cnt_duplicate_subgoal/steps)
				if have_nouse:
					nouse_s[episode_id].append(cnt_nouse_subgoal/2/steps)

		for episode_id in test_task:
			item = {'S': S[episode_id], 'L': L[episode_id], 'L_remove_comm':L_remove_comm[episode_id], 'step_data': step_data[episode_id], 'duplicate_subgoal': dup_s[episode_id], 'influence': cnt_influences[episode_id], 'env_id': env_id[episode_id]}
			if have_nouse:
				item['nouse_subgoal'] = nouse_s[episode_id]
			test_results[episode_id] = item

		pickle.dump(test_results, open(results_path, 'wb'))
		# print('average steps (finishing the tasks):', np.array(steps_list).mean() if len(steps_list) > 0 else None)
		# print('failed_tasks:', failed_tasks)

#test_speedup.py---------------------------------------------------------------


def means_in_num_runs(l):
	if len(l) < num_runs:
		num = len(l)
		# print(f"only {num} runs", end='')
	else:
		num = num_runs
	return np.mean([l[i] for i in range(num)])

def get_success_rate(result, single_result, episode, step_len_type):
	return means_in_num_runs(result[episode]['S'])

# def get_step_number(result, single_result, episode, step_len_type):
# return means_in_num_runs(result[episode][step_len_type])

def get_speedup(result, single_result, episode, step_len_type):
	return means_in_num_runs(single_result[episode][step_len_type]) / means_in_num_runs(result[episode][step_len_type]) - 1

def get_EI(result, single_result, episode, step_len_type):
	return 1 - means_in_num_runs(result[episode][step_len_type]) / means_in_num_runs(single_result[episode][step_len_type])

def get_reward(result, single_result, episode, step_len_type):
	return means_in_num_runs(result[episode]['S']) - 0.004 * means_in_num_runs(result[episode][step_len_type])

def get_duplicate_subgoal(result, single_result, episode, step_len_type):
	return means_in_num_runs(result[episode]['duplicate_subgoal'])

def get_nouse_subgoal(result, single_result, episode, step_len_type):
	return means_in_num_runs(result[episode]['nouse_subgoal'])

def get_influence_rate(result, single_result, episode, step_len_type):
	return means_in_num_runs(result[episode]['influence'])

def get_step_number(result, single_result, episode, step_len_type):
	if 'step_data' in result[episode].keys() and step_len_type in ['S', 'B', 'E', 'None', 'others']:
		return np.mean([result[episode]['step_data'][j][step_len_type] for j in range(num_runs)])
	if step_len_type == 'L_remove_comm':
		return means_in_num_runs(result[episode]['L_remove_comm'])
	else:
		return means_in_num_runs(result[episode]['L'])

def analyse():
	def filter(result):
		new_result = {}
		for key, val in result.items():
			if key not in test_task:
				continue
			new_result[key] = {}
			for k, vl in val.items():
				if k !='L':
					continue
				new_result[key][k] = vl[:num_runs]
		return new_result
	def make_means_in_episode(result, single_result, step_len_type):
		def means_in_episode(f):
			return np.mean([f(result, single_result, i, step_len_type) for i in test_task])
		return means_in_episode
	def make_std_in_episode(result, single_result, step_len_type):
		def std_in_episode(f):
			return np.std([f(result, single_result, i, step_len_type) for i in test_task])
		return std_in_episode

	for record_dir, method_names in zip(args.record_dir, args.method_names):
		results_path = os.path.join(record_dir, args.results_pik_name)
		a = pickle.load(open(results_path, 'rb'))
		print('-------------------------------------------------------')
		print('name', method_names)

		if args.verbose:
			print(f"single: {json.dumps(filter(single_result))}")
			print(f"{method_names}: {json.dumps(filter(a))}")

		print(f'{num_per_task = }, {num_runs = }')

		print('average success rate =', make_means_in_episode(a, None, None)(get_success_rate))
		if args.analyse_std:
			print('std success rate =', make_std_in_episode(a, None, None)(get_success_rate))

		pie_data = {k : [] for k in pie_chart_keys}
		for episode_id in test_task:
			for i in range(min(num_runs, len(a[episode_id]['step_data']))):
				for k in pie_chart_keys:
					pie_data[k].append(a[episode_id]['step_data'][i][k])
		for k,v in pie_data.items():
			pie_data[k] = list(np.array(v).mean(axis=0))
		# print(pie_data)
		print('comm:', pie_data['comm'])

		for step_len_type in ['L', 'L_remove_comm']:
			print('-------------------------------')
			if step_len_type not in a[test_task[0]].keys():
				print(f"Unavailable {step_len_type}\n")
				continue
			if 'comm' not in method_names and step_len_type == 'L_remove_comm':
				continue
			print(f'{step_len_type = }')
			means_in_episode = make_means_in_episode(a, single_result, step_len_type)
			std_in_episode = make_std_in_episode(a, single_result, step_len_type)



			#special, not added to plot
			# print('average success step number =', np.mean([a[i][step_len_type][j] for i in test_task for j in range(num_runs) if a[i]['S'][j]]))



			print('average step number =', '%.3g' % means_in_episode(get_step_number))
			if args.analyse_std:
				print('std step number =', '%.3g' % std_in_episode(get_step_number))
			# if 'cnt_subgoal' in a[0].keys():
			# 	#cnt_duplicate_subgoal, cnt_nouse_subgoal
			# 	print('average cnt duplicate subgoal =', np.mean([a[i]['cnt_subgoal'][j][0] for i in test_task for j in range(num_runs)]))
			# 	print('average cnt nouse subgoal =', np.mean([a[i]['cnt_subgoal'][j][1] for i in test_task for j in range(num_runs)]))


			# su = [np.mean([single[i][step_len_type][j] / a[i][step_len_type][j] - 1 for i in test_task]) for j in range(num_runs)]

			# if step_len_type == 'L_comm':
			# continue
			print("speedup =", '%.3g' % means_in_episode(get_speedup))
			if args.analyse_std:
				print('std speedup =', '%.3g' % std_in_episode(get_speedup))
			# print(f"std speedup = {np.std([np.mean(single[i][step_len_type]) / np.mean(a[i][step_len_type]) - 1 for i in test_task])}")

			# filtered_EI_values = [value for value in EI_values if value is not None]


			print("EI =", '%.3g' % means_in_episode(get_EI))
			if args.analyse_std:
				print('std EI =', '%.3g' % std_in_episode(get_EI))
			# print(f"std EI = {np.std([1 - np.mean(a[i][step_len_type]) / np.mean(single[i][step_len_type]) for i in test_task])}")
			# print('std EI =', np.std(filtered_EI_values))



			# print('list accumulated reward =', ar)
			print('accumulated reward =', '%.3g' % means_in_episode(get_reward))
			if args.analyse_std:
				print('std accumulated reward =', '%.3g' % std_in_episode(get_reward))
			# std_reward = np.std(ar)
			# print('std accumulated reward =', std_reward)

			if args.subgoal_analysis:
				print('duplicate subgoal =', '%.3g%%' % (100 * means_in_episode(get_duplicate_subgoal)))

				if 'nouse_subgoal' in a[0]:
					print('no use subgoal =', '%.3g%%' % (100 * means_in_episode(get_nouse_subgoal)))

				print('influence rate =', '%.3g%%' % (100 * means_in_episode(get_influence_rate)))

#plot.py--------------------------------------------------------------

def make_data_to_plot():
	if args.remove_comm_action:
		assert args.action_type == 'L'
	step_len_type = 'L' if not args.remove_comm_action else 'L_remove_comm'
	if args.action_type != 'L':
		step_len_type = args.action_type
	# all func in dict are metric args that can be used in utils_plot.py
	calc_func = {'Success Rate' : get_success_rate, 'Step Number' : get_step_number, 'Speed Up' : get_speedup, 'EI' : get_EI, 'Reward' : get_reward, 'Duplicate Subgoal Rate' : get_duplicate_subgoal, 'No Use Subgoal Rate' : get_nouse_subgoal, 'Influence Rate': get_influence_rate}
	test_results = []

	for record_dir, method_names in zip(args.record_dir, args.method_names):
		results_path = os.path.join(record_dir, args.results_pik_name)
		result = pickle.load(open(results_path, 'rb'))
		for episode_id in test_task:
			item = {'episode_id': episode_id, 'task_name' : env_task_set[episode_id]['task_name'], 'env_id':env_task_set[episode_id]['env_id'], 'method' : method_names}
			for k, v in calc_func.items():
				if k != args.metric:
					continue
				item[k] = v(result, single_result, episode_id, step_len_type)
			test_results.append(item)
	# total_copy = deepcopy(test_results)
	# for item in total_copy:
	#     item['task_name'] = 'total'
	# test_results.extend(total_copy)
	return test_results

def make_pie_data_and_plot():
	pie_datas = []
	workingpwd = os.getcwd()
	for record_dir, method_names in zip(args.record_dir, args.method_names):
		os.chdir(workingpwd)
		pie_data = {k:0 for k in pie_chart_plot_keys}
		results_path = os.path.join(record_dir, args.results_pik_name)
		result = pickle.load(open(results_path, 'rb'))
		for episode_id in test_task:
			for i in range(num_runs):
				for k in pie_chart_plot_keys:
					pie_data[k] += sum(result[episode_id]['step_data'][i][k])
		# remove value 0
		colors = sns.color_palette('pastel')[0:len(pie_data)]
		colors = [i for i,v in zip(colors, pie_data.values()) if v != 0]
		pie_data = {k:v for k,v in pie_data.items() if v != 0}
		pie_datas.append(pie_data)
		plt.pie(list(pie_data.values()), labels=list(pie_data.keys()), autopct='%1.1f%%', startangle=90, colors=colors)
		plt.title(method_names)
		os.chdir(pwd)
		os.chdir('Output')
		print(f'{method_names} pie.png')
		plt.savefig(f'{method_names} pie.png')
		plt.close()
	os.chdir(workingpwd)
	return pie_datas

def plot(test_results = None):
	palette = ["#b3b3b3", "#fff08a", "#83deed", "#8ad09d", "#cc95fa", "#f6958f"]
	workingpwd = os.getcwd()
	os.chdir(pwd)
	os.chdir(args.output_dir)
	df = pd.DataFrame(test_results)
	sns.set_theme(style="whitegrid")
	# penguins = sns.load_dataset("penguins")
	# Draw a nested barplot by species and sex

	def merge_show():
		g = sns.catplot(
			data=df, kind="bar",
			x="task_name", y=args.metric, hue="method",
			errorbar="sd" if args.errorbar else None, palette=palette, alpha=.6, height=6
		)
		g.set(ylim=args.ylim)
		ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]
		for c in ax.containers:
			labels = [f'{round(v.get_height()) if args.method_names == "StepNumber" else (v.get_height()):.2f}' for v in c]
			ax.bar_label(c, labels=labels, label_type='edge')

		g.despine(left=True)
		g.set_axis_labels("", args.metric)
		g.set_xticklabels(ax.get_xticklabels(),rotation = 20)
		g.legend.set_title("")

		#put hue from right to the top
		g.legend.remove()
		plt.legend(title="Method", bbox_to_anchor=(0.5, 1.15), loc="center", ncol=args.ncol)
		plt.gcf().subplots_adjust(top=0.8)
		# plt.show()
		print('save to', f'{"remove_comm_" if args.remove_comm_action else ""}{args.metric}_plot.png')
		plt.savefig(f'{"remove_comm_" if args.remove_comm_action else ""}{args.metric}_plot.png')
		plt.close()

	def split_show():
		grouped = df.groupby('env_id')
		# create an empty list to save grouped tables
		grouped_tables = []

		for name, group in grouped:
			grouped_tables.append(group)
		for plotdf in grouped_tables:
			g = sns.catplot(
				data=plotdf, kind="bar",
				x="task_name", y=args.metric, hue="method",
				errorbar="sd" if args.errorbar else None, palette=palette, alpha=.6, height=6
			)
			g.set(ylim=args.ylim)
			ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]
			for c in ax.containers:
				labels = [f'{round(v.get_height()) if args.method_names == "StepNumber" else (v.get_height()):.2f}' for v in c]
				ax.bar_label(c, labels=labels, label_type='edge')

			g.despine(left=True)
			g.set_axis_labels("", args.metric)
			g.set_xticklabels(ax.get_xticklabels(),rotation = 20)
			g.legend.set_title("")
			#put hue from right to the top
			g.legend.remove()
			plt.legend(bbox_to_anchor=(0.5, 1.15), loc="center", ncol=args.ncol)
			env_id = plotdf.iloc[0]['env_id']
			plt.title(f"Apt.{env_id}")
			plt.gcf().subplots_adjust(left=args.left, right=args.right, bottom=args.bottom, top=args.top)
			print('save to', f'{"remove_comm_" if args.remove_comm_action else ""}{args.metric}_{env_id}.png')
			plt.savefig(f'{"remove_comm_" if args.remove_comm_action else ""}{args.metric}_{env_id}.png')
			plt.close()
	print('-------------------------------------------------------')
	print('plotting...')
	if not args.split_show:
		merge_show()
	else:
		split_show()
	os.chdir(workingpwd)


def print_log(a, output_dir, task_id):
	with open(output_dir, "w") as f:
		sys.stdout = f
		print(a.keys())
		print('task_goal: ', a['goals'][0])
		print('total_goal: ', env_task_set[task_id]['total_goal'])
		print('steps: ', len(a['action'][0]))
		if len(a['action'][1]) >= 1:
			action = {x: ('' if a['action'][0][x] is None else a['action'][0][x]) + '|' + ('' if a['action'][1][x] is None else a['action'][1][x]) for x in range(len(a['action'][0]))}
			plan = {x: ('' if a['plan'][0][x] is None else a['plan'][0][x][0] if type(a['plan'][0][x]) is list else a['plan'][0][x]) + '|' + ('' if a['plan'][1][x] is None else a['plan'][1][x]) for x in range(len(a['plan'][0]))}
		else:
			action = {x: ('' if a['action'][0][x] is None else a['action'][0][x]) for x in range(len(a['action'][0]))}
			plan = {x: ('' if a['plan'][0][x] is None else a['plan'][0][x][0] if type(a['plan'][0][x]) is list else a['plan'][0][x]) for x in range(len(a['plan'][0]))}
		print(json.dumps({'action': action, 'plan': plan}, indent=4))
		if 'LLM' in a.keys():
			for agent_id in args.agent_id:
				print(f"agent: {agent_id}\n")
				for num, o in enumerate(a['LLM'][agent_id]):
					print(json.dumps(o, indent=4))

		if args.verbose:
			for agent_id in args.agent_id:
				print(f"agent: {agent_id}\n")
				if len(a['obs'][agent_id]) != len(a['action'][agent_id]):
					continue
				for step, action in enumerate(a['action'][agent_id]):
					print(f"step: {step}\n")
					o = {
						'action': action,
						'plan': a['plan'][agent_id][step],
						'graph': a['graph'][agent_id][step],
						'obs': a['obs'][agent_id][step],
						# 'pos': [x["obj_transform"]["position"] for x in a['graph'][agent_id][step]['nodes'] if x['id'] == agent_id + 1],
					}
					print(json.dumps(o, indent=4))


	# o = {'goals': a['goals'],
	# 	 'action': a['action'],
	# 	 'prompts': a['prompts'],
	# 	 'outputs': a['outputs'],
	# 	 'cot_outputs': a['cot_outputs'],
	# 	 'plan': a['plan'],
	# 	 'finished': a['finished'],
	# 	 'graph': a['graph'][1],
	# 	 'obs': a['obs'][1],
	# 	 'pos': [x["obj_transform"]["position"] for g in a['graph'][1] for x in g['nodes'] if x['id'] == 2]
	# 	 # 'graph': a['graph'][1],
	# 	 }
	# print(json.dumps(o, indent=4))
def pretty_print_pickle_logs():
	if args.current_log:
		a = pickle.load(open(os.path.join(args.record_dir[0], 'log.pik'), 'rb'))
		print_log(a, 'action.log', 0)
	else:
		for record_dir in args.record_dir:
			mode = record_dir.split("/")[-1]
			if mode == "":
				mode = record_dir.split("/")[-2]
			output_dir = os.path.join(args.output_dir, mode)
			Path(output_dir).mkdir(parents=True, exist_ok=True)
			for task_id in args.test_task:
				for seed in range(args.num_runs):
					a = pickle.load(open(os.path.join(record_dir, f'logs_agent_{task_id}_{task_names[task_id // 10]}_{seed}.pik'), "rb"))
					print_log(a, os.path.join(output_dir, f"{task_id}_{seed}.log"), task_id)
				# s = pickle.load(open(os.path.join(args.single_dir, f'logs_agent_{task_id}_{task_names[task_id // 20]}_{args.num_runs}.pik', "rb")))


if __name__ == '__main__':
	if args.print_log:
		pretty_print_pickle_logs()
	if args.generate_result:
		generate_result()
	if args.analyse or args.plot:
		single_result = pickle.load(open(os.path.join(args.single_dir, args.results_pik_name), 'rb'))
	if args.analyse:
		analyse()
	test_results = None
	if args.plot:
		test_results = make_data_to_plot()
	if args.plot:
		plot(test_results)
	if args.plotpie:
		make_pie_data_and_plot()