lm_id=Llama-2-13b-chat-hf
port=10691
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--run_id $lm_id \
--port $port \
--agents lm_agent \
--source hf \
--lm_id /mnt/nfs/share/Llama/${lm_id} \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--screen_size 256 \
--debug

pkill -f -9 "port $port"