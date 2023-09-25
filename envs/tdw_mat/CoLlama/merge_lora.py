from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
from peft import PeftModel
import argparse
from datasets import load_dataset
from typing import (
	Callable,
	Dict,
	Optional,
	Sequence,
	Union,
	Mapping,
	Any,
	List,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

def tokenize_dialog(datapoint):
		dialog = datapoint['dialog']
		B_INST, E_INST = "[INST]", "[/INST]"
		B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
		prompt_tokens = []
		# print(dialog)
		if dialog[0]["role"] == "system":
			dialog = [
						 {
							 "role": dialog[1]["role"],
							 "content": B_SYS
										+ dialog[0]["content"]
										+ E_SYS
										+ dialog[1]["content"],
						 }
					 ] + dialog[2:]
		assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
			[msg["role"] == "assistant" for msg in dialog[1::2]]
		), (
			"model only supports 'system', 'user' and 'assistant' roles, "
			"starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
		)
		assert (
				dialog[-1]["role"] == "assistant"
		), f"Last message must be from assistant for fine-tuning, got {dialog[-1]['role']}"
		dialog_tokens: List[int] = sum(
			[
				[tokenizer.bos_token_id] +
				tokenizer.encode(
					f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
					add_special_tokens=False
				)
				+ [tokenizer.eos_token_id]
				for prompt, answer in zip(dialog[::2], dialog[1::2], )
			],
			[],
		)
		# assert len(dialog_tokens) <= cutoff_len, f"too long! {len(dialog_tokens)} > {cutoff_len}"
		# prompt_tokens.append(dialog_tokens)
		prompt_tokens = dialog_tokens

		user_tokens = [tokenizer.bos_token_id] + tokenizer.encode(
				f"{B_INST} {dialog[0]['content'].strip()} {E_INST}", add_special_tokens=False)
		user_prompt_len = len(user_tokens)
		return {"input_ids": prompt_tokens, "attention_mask": [1] * len(dialog_tokens), "labels": [-100] * user_prompt_len + prompt_tokens.copy()[user_prompt_len:]}


def prepare_inputs(
	data: Union[torch.Tensor, Any], device: Union[str, int, torch.device]
) -> Union[torch.Tensor, Any]:
	if isinstance(data, Mapping):
		return type(data)(
			{k: prepare_inputs(v, device) for k, v in data.items()}
		)  # noqa
	elif isinstance(data, (tuple, list)):
		return type(data)(prepare_inputs(v, device) for v in data)
	elif isinstance(data, torch.Tensor):
		return data.to(device)  # This can break with deepspeed.
	return data


@torch.no_grad()
def Eval(model, eval_dataloader, dev):
	model.eval()
	val_loss = 0
	for i, batch in tqdm(enumerate(eval_dataloader)):
		if i > 20:
			break
		batch = prepare_inputs(batch, dev)
		outputs = model(**batch)
		val_loss += outputs.loss.detach().float()
	val_loss /= len(eval_dataloader)
	return val_loss


parser = argparse.ArgumentParser()

parser.add_argument("--base_model", type=str, default='/mnt/nfs/share/Llama/Llama-2-13b-chat-hf')
parser.add_argument("--lora_weights", type=str, default='/mnt/nfs/share/Llama/Collama/Collama-13b-chat-lora/lora')
parser.add_argument("--save_dir", type=str, default='/mnt/nfs/share/Llama/Collama/Collama-13b-chat-lora')
args = parser.parse_args()
model = LlamaForCausalLM.from_pretrained(
			args.base_model,
			torch_dtype=torch.float16,
			device_map={"": "cpu"},
		)
lora_model = PeftModel.from_pretrained(
	model,
	args.lora_weights,
	is_trainable=False,
)

lora_model = lora_model.merge_and_unload()
lora_model.save_pretrained(args.save_dir)

print(f"merged and saved to {args.save_dir}")

model = LlamaForCausalLM.from_pretrained(
			args.lora_weights,
			device_map='auto',
			load_in_4bit=True,
		)
tokenizer = LlamaTokenizer.from_pretrained(args.save_dir)
data_collator = transformers.DataCollatorForSeq2Seq(
	tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)
data = load_dataset("json", data_files='data_good.json')["train"].map(tokenize_dialog)
data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataloader = DataLoader(
	data, shuffle=False,
	collate_fn=data_collator,
	batch_size=4
)

val_loss = Eval(model, eval_dataloader, 'cuda')
print(val_loss)