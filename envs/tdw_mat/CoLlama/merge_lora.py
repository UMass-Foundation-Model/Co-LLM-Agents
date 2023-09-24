from transformers import LlamaForCausalLM
import torch
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--base_model", type=str, default='/gpfs/u/home/AICD/AICDhnng/scratch-shared/llama_zf/llama-2-13b-chat-hf')
parser.add_argument("--lora_weights", type=str, default='/gpfs/u/home/AICD/AICDhnng/scratch/Collama/Collama-13b-chat-lora')
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
lora_model.save_pretrained(args.lora_weights)