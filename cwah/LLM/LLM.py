import random

import openai
import torch
import json
import os
import pandas as pd
from openai.error import OpenAIError
import backoff


class LLM:
	def __init__(self,
				 source,  # 'huggingface' or 'openai'
				 lm_id,
				 prompt_template_path,
				 communication,
				 cot,
				 sampling_parameters,
				 agent_id
				 ):
		self.goal_desc = None
		self.goal_location_with_r = None
		self.agent_id = agent_id
		self.agent_name = "Alice" if agent_id == 1 else "Bob"
		self.oppo_name = "Alice" if agent_id == 2 else "Bob"
		self.oppo_pronoun = "she" if agent_id == 2 else "he"
		self.debug = sampling_parameters.debug
		self.goal_location = None
		self.goal_location_id = None
		self.roomname2id = {}
		self.rooms = []
		self.prompt_template_path = prompt_template_path
		self.single = 'single' in self.prompt_template_path
		df = pd.read_csv(self.prompt_template_path)
		self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
		if communication:
			self.generator_prompt_template = df['prompt'][1].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
		else:
			self.generator_prompt_template = None

		self.communication = communication
		self.cot = cot
		self.source = source
		self.lm_id = lm_id
		self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id
		self.OPENAI_KEY = None
		self.total_cost = 0
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		if self.source == 'openai':
			openai.api_key = os.getenv("OPENAI_KEY")
			if self.chat:
				self.sampling_params = {
					"max_tokens": sampling_parameters.max_tokens,
					"temperature": sampling_parameters.t,
					"top_p": sampling_parameters.top_p,
					"n": sampling_parameters.n,
				}
			else:
				self.sampling_params = {
					"max_tokens": sampling_parameters.max_tokens,
					"temperature": sampling_parameters.t,
					"top_p": sampling_parameters.top_p,
					"n": sampling_parameters.n,
					"logprobs": sampling_parameters.logprobs,
					"echo": sampling_parameters.echo,
				}
		elif source == 'huggingface':
			self.sampling_params = {
				"max_new_tokens": sampling_parameters.max_tokens,
				"temperature": sampling_parameters.t,
				"top_p": sampling_parameters.top_p,
				"num_return_sequences": sampling_parameters.n,
				'use_cache': True,
				# 'output_scores': True,
				'return_dict_in_generate': True,
				'do_sample': True,
				'early_stopping': True,
			}
		elif source == "debug":
			self.sampling_params = sampling_parameters
		else:
			raise ValueError("invalid source")

		def lm_engine(source, lm_id, device):
			if source == 'huggingface':
				from transformers import AutoModelForCausalLM, AutoTokenizer, LLaMATokenizer, LLaMAForCausalLM
				print(f"loading huggingface model {lm_id}")
				if 'llama' in lm_id or 'alpaca' in lm_id:
					tokenizer = LLaMATokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce') # '/gpfs/u/scratch/AICD/AICDhnng/.cache')
					model = LLaMAForCausalLM.from_pretrained(lm_id, # device_map="balanced_low_0",
															 # max_memory = {0: "10GB", 1: "20GB", 2: "20GB", 3: "20GB",4: "20GB",5: "20GB",6: "20GB",7: "20GB"},
															 torch_dtype=torch.float16, low_cpu_mem_usage=True,
        														load_in_8bit=False,
															 cache_dir='/work/pi_chuangg_umass_edu/.cahce')\
																.to(device)
				else:
					tokenizer = AutoTokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce')
					model = AutoModelForCausalLM.from_pretrained(lm_id, torch_dtype=torch.float16,
																 pad_token_id=tokenizer.eos_token_id,
																 cache_dir='/work/pi_chuangg_umass_edu/.cahce').to(
						device)
				print(f"loaded huggingface model {lm_id}")

			@backoff.on_exception(backoff.expo, OpenAIError)
			def _generate(prompt, sampling_params):
				usage = 0
				if source == 'openai':
					try:
						if self.chat:
							response = openai.ChatCompletion.create(
								model=lm_id, messages=prompt, **sampling_params
							)
							# print(json.dumps(response, indent=4))
							if self.debug:
								with open(f"LLM/chat_raw.json", 'a') as f:
									f.write(json.dumps(response, indent=4))
									f.write('\n')
							generated_samples = [response['choices'][i]['message']['content'] for i in
												 range(sampling_params['n'])]
							if 'gpt-4' in self.lm_id:
								usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage']['completion_tokens'] * 0.06 / 1000
							elif 'gpt-3.5' in self.lm_id:
								usage = response['usage']['total_tokens'] * 0.002 / 1000
						# mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
						# 				  range(sampling_params['n'])]
						elif "text-" in lm_id:
							response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
							# print(json.dumps(response, indent=4))
							if self.debug:
								with open(f"LLM/raw.json", 'a') as f:
									f.write(json.dumps(response, indent=4))
									f.write('\n')
							generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
						# mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
						# 			  range(sampling_params['n'])]
						else:
							raise ValueError(f"{lm_id} not available!")
					except OpenAIError as e:
						print(e)
						raise e
				elif source == 'huggingface':
					input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
					prompt_len = input_ids.shape[-1]
					# print(sampling_params)
					output_dict = model.generate(input_ids, # max_length=prompt_len + sampling_params['max_new_tokens'],
												 **sampling_params)
					generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
					# vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)
					# token_log_probs = torch.gather(vocab_log_probs, 2,
					# 							   output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()
					for i, sample in enumerate(generated_samples):
						stop_idx = sample.index('\n') if '\n' in sample else None
						generated_samples[i] = sample[:stop_idx]
					# 	token_log_probs[i] = token_log_probs[i][:stop_idx]
					# mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
				elif source == "debug":
					return ["navigation"]
				else:
					raise ValueError("invalid source")
				# generated_samples = [sample.strip().lower() for sample in generated_samples]
				return generated_samples, usage

			return _generate

		self.generator = lm_engine(self.source, self.lm_id, self.device)


	def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
		self.rooms = rooms_name
		self.roomname2id = roomname2id
		self.goal_location = goal_location
		self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
		self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)


	def goal2description(self, goals, goal_location_room):  # {predicate: count}
		# print(goals)
		map_rel_to_pred = {
			'inside': 'into',
			'on': 'onto',
		}
		s = "Find and put "
		r = None
		for predicate, vl in goals.items():
			relation, obj1, obj2 = predicate.split('_')
			count = vl
			if count == 0:
				continue
			if relation == 'holds':
				continue
				# s += f"Alice holds a book, "
			elif relation == 'sit':
				continue
				# s += f"Alice sits in {obj2}, "
			else:
				s += f"{count} {obj1}{'s' if count > 1 else ''}, "
				r = relation
		if r is None:
			return "None."

		s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."
		# if type(goal_location_room) is not list:
		# 	s += f" in the {goal_location_room}."
		# else:
		# 	ss = ' or '.join([f'{room}' for room in goal_location_room])
		# 	s += f", which may be in the {ss}."
		return s, f"{map_rel_to_pred[r]} the {self.goal_location}"


	# def get_obj(self, obs, text, k=1):
	# 	id2node = {node['id']: node for node in obs['nodes']}
	# 	cnt = 0
	# 	for x, node in id2node.items():
	# 		if f'({x})' in text:
	# 			cnt += 1
	# 			if cnt != k: continue
	# 			return f"<{node['class_name']}> ({x})"
	# 	print("WARNING! No object correctly parsed!!! Random choose one")
	# 	x, node = random.choice(list(id2node.items()))
	# 	return f"<{node['class_name']}> ({x})"
	#
	#
	# def get_action(self, obs, text):
	# 	if '[open]' in text or '[close]' in text or '[grab]' in text or '[walktowards]' in text:
	# 		return f"[{text.split(']')[0].split('[')[-1]}] {self.get_obj(obs, text)}"
	# 	elif 'putback' in text or 'putin' in text:
	# 		obj1 = self.get_obj(obs, text)
	# 		obj2 = self.get_obj(obs, text, 2)
	# 		return f"[{text.split(']')[0].split('[')[-1]}] {obj1} {obj2}"

	def parse_answer(self, available_actions, text):
		for i in range(len(available_actions)):
			action = available_actions[i]
			if action in text:
				return action

		for i in range(len(available_actions)):
			action = available_actions[i]
			option = chr(ord('A') + i)
			# txt = text.lower()
			if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
				return action
		print("WARNING! Fuzzy match!")
		for i in range(len(available_actions)):
			action = available_actions[i]
			if self.communication and i == 0:
				continue
			act, name, id = action.split(' ')
			option = chr(ord('A') + i)
			if f"{option} " in text or act in text or name in text or id in text:
				return action
		print("WARNING! No available action parsed!!! Random choose one")
		return random.choice(available_actions)


	def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
		sss = {}
		for room, objs in ungrabbed_objects.items():
			cons = unchecked_containers[room]
			extra_obj = None
			if type(goal_location_room) is not list and goal_location_room == room:
				extra_obj = self.goal_location
			if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
				sss[room] = f"The {room} is unexplored. "
				continue
			s = ""
			s_obj = ""
			s_con = ""
			if extra_obj is not None:
				s_obj = f"{extra_obj}, "
			if objs is not None and len(objs) > 0:
				if len(objs) == 1:
					x = objs[0]
					s_obj += f"<{x['class_name']}> ({x['id']})"
				else:
					ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
					s_obj += ss
			elif extra_obj is not None:
				s_obj = s_obj[:-2]
			if cons is not None and len(cons) > 0:
				if len(cons) == 1:
					x = cons[0]
					s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
				else:
					ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
					s_con = f"unchecked containers " + ss
			if s_obj == "" and s_con == "":
				s += 'nothing'
				if room_explored is not None and not room_explored[room]:
					s += ' yet'
			elif s_obj != "" and s_con != "":
				s += s_obj + ', and ' + s_con
			else:
				s += s_obj + s_con
			sss[room] = s

		if len(satisfied) == 0:
			s = ""
		else:
			s = f"{'I' if self.single else 'We'}'ve already found and put "
			s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
			s += ' ' + self.goal_location_with_r + '. '

		if len(grabbed_objects) == 0:
			s += "I'm holding nothing. "
		else:
			s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
			if len(grabbed_objects) == 2:
				s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
		s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
		### opponent modeling
		if not self.single:
			ss = ""
			if len(opponent_grabbed_objects) == 0:
				ss += "nothing. "
			else:
				ss += f"<{opponent_grabbed_objects[0]['class_name']}> ({opponent_grabbed_objects[0]['id']}). "
				if len(opponent_grabbed_objects) == 2:
					ss = ss[:-2] + f" and <{opponent_grabbed_objects[1]['class_name']}> ({opponent_grabbed_objects[1]['id']}). "
			if opponent_last_room is None:
				s += f"I don't know where {self.oppo_name} is. "
			elif opponent_last_room == current_room['class_name']:
				s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
			else:
				s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

		for room in self.rooms:
			if room == current_room['class_name']:
				continue
			if 'unexplored' in sss[room]:
				s += sss[room]
			else:
				s += f"I found {sss[room]} in the {room}. "

		return s


	def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored):
		"""
		[goexplore] <room>
		[gocheck] <container>
		[gograb] <target object>
		[goput] <goal location>
		[send_message] <"">
		"""
		available_plans = []
		if self.communication and message is not None:
			available_plans.append(f"[send_message] <{message}>")
		for room in self.rooms:
			if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
				continue
			available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
		if len(grabbed_objects) < 2:
			for cl in unchecked_containers.values():
				if cl is None:
					continue
				for container in cl:
					available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
			for ol in ungrabbed_objects.values():
				if ol is None:
					continue
				for obj in ol:
					available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
		if len(grabbed_objects) > 0:
			available_plans.append(f"[goput] {self.goal_location}")
		
		plans = ""
		for i, plan in enumerate(available_plans):
			plans += f"{chr(ord('A') + i)}. {plan}\n"

		return plans, len(available_plans), available_plans

			
	def run(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):
		info = {}
		# goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
		progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
		action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
		dialogue_history_desc = '\n'.join(dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history)
		prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
		prompt = prompt.replace('$PROGRESS$', progress_desc)
		prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)
		message = None

		if self.communication:
			prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
			if not action_history[-1].startswith('[send_message]'):
				gen_prompt = self.generator_prompt_template.replace('$GOAL$', self.goal_desc)
				gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
				gen_prompt = gen_prompt.replace('$ACTION_HISTORY$', action_history_desc)
				gen_prompt = gen_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
				gen_prompt = gen_prompt + f"\n{self.agent_name}:"
				chat_prompt = [{"role": "user", "content": gen_prompt}]
				outputs, usage = self.generator(chat_prompt if self.chat else gen_prompt, self.sampling_params)
				self.total_cost += usage
				message = outputs[0]
				info['message_generator_prompt'] = gen_prompt
				info['message_generator_outputs'] = outputs
				info['message_generator_usage'] = usage
				if self.debug:
					print(f"message_generator_prompt:\n{gen_prompt}")
					print(f"message_generator_outputs:\n{message}")

		available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored)
		if num == 0 or (message is not None and num == 1):
			print("Warning! No available plans!")
			plan = None
			info.update({"num_available_actions": num,
					 "plan": None})
			return plan, info

		prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

		if self.cot:
			prompt = prompt + " Let's think step by step."
			if self.debug:
				print(f"cot_prompt:\n{prompt}")
			chat_prompt = [{"role": "user", "content": prompt}]
			outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
			output = outputs[0]
			self.total_cost += usage
			info['cot_outputs'] = outputs
			info['cot_usage'] = usage
			if self.debug:
				print(f"cot_output:\n{output}")
			chat_prompt = [{"role": "user", "content": prompt},
						   {"role": "assistant", "content": output},
						   {"role": "user", "content": "Answer with only one best next action. So the answer is"}]
			normal_prompt = prompt + output + ' So the answer is'
			outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
			output = outputs[0]
			self.total_cost += usage
			info['output_usage'] = usage
			if self.debug:
				print(f"base_output:\n{output}")
				print(f"total cost: {self.total_cost}")
		else:
			if self.debug:
				print(f"base_prompt:\n{prompt}")
			outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt, self.sampling_params)
			output = outputs[0]
			info['cot_usage'] = usage
			if self.debug:
				print(f"base_output:\n{output}")
		plan = self.parse_answer(available_plans_list, output)
		if self.debug:
			print(f"plan: {plan}\n")
		info.update({"num_available_actions": num,
					 "prompts": prompt,
					 "outputs": outputs,
					 "plan": plan,
					 "total_cost": self.total_cost})
		return plan, info

