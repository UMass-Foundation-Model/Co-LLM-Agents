lm_id=Llama-2-13b-hf
port=1091
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--run_id $lm_id \
--port $port \
--agents lm_agent \
--source hf \
--lm_id /mnt/gluster/home/chenpeihao/Projects/Co-LLM-Agents/envs/tdw_mat/Llama-2-13b-hf \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--debug

pkill -f -9 "port $port"