import pandas as pd
import json

df = pd.read_csv("Alice_0921.csv")
df2 = pd.read_csv("Bob_0921.csv")
df = pd.concat([df, df2], ignore_index=True)
df = df.sort_values(by='id')
quality_comm = {}
quality_COT = {}
quality_plan = {}
for index, row in df.iterrows():
	quality_comm[row['id']] = row['quality_comm[bad, fine, good]']
	quality_COT[row['id']] = row['quality_COT[bad, fine, good]']
	quality_plan[row['id']] = row['quality_final_plan[bad, fine, good]']

df_prompt = pd.read_csv("LLM_data_Alice.csv")
df_prompt_2 = pd.read_csv("LLM_data_Bob.csv")
df_prompt = pd.concat([df_prompt, df_prompt_2], ignore_index=True)
df_prompt = df_prompt.sort_values(by='id')
data_good = []
data_unfiltered = []
for index, row in df_prompt.iterrows():
	if not pd.isna(row['prompt_comm']):
		if row['id'] in quality_comm and quality_comm[row['id']] == 'good':
			data_good.append({"index": f"comm_{row['episode']}_{row['agent']}_{row['id']}","dialog": [{"role": "user", "content": row['prompt_comm']}, {"role": "assistant", "content": row["output_comm"]}]})
		elif not (row['id'] in quality_comm and quality_comm[row['id']] == 'bad'):
			data_unfiltered.append({"index": f"comm_{row['episode']}_{row['agent']}_{row['id']}", "dialog": [{"role": "user", "content": row['prompt_comm']}, {"role": "assistant", "content": row["output_comm"]}]})
	if row['id'] in quality_COT and quality_COT[row['id']] == 'good' and (quality_plan[row['id']] != 'bad'):
		data_good.append({"index": f"cot_{row['episode']}_{row['agent']}_{row['id']}","dialog": [{"role": "user", "content": row['prompt_plan']}, {"role": "assistant", "content": row["output_plan_stage_1"].strip()}]})
	elif not (row['id'] in quality_COT and quality_COT[row['id']] == 'bad'):
		data_unfiltered.append({"index": f"cot_{row['episode']}_{row['agent']}_{row['id']}","dialog": [{"role": "user", "content": row['prompt_plan']}, {"role": "assistant", "content": row["output_plan_stage_1"].strip()}]})

print(f"data_good: {len(data_good)}")
with open("data_good.json", "w") as f:
	json.dump(data_good, f, indent=4)

print(f"data_unfiltered: {len(data_unfiltered)}")
with open("data_unfiltered.json", "w") as f:
	json.dump(data_unfiltered, f, indent=4)

# with open("data_good.json", "r") as f:
# 	re = json.load(f)
# 	print(re)
# df.to_csv("merged.csv", index=False)