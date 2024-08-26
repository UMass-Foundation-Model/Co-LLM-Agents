lm_id=Llama-2-13b-chat-hf
port=10005
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--source hf \
--lm_id /mnt/nfs/share/Llama/${lm_id} \
--run_id LMs-$lm_id \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 11 17 18 1 2 3 21 22 23 4 5 6 7 8 9 10 12 13 14 15 16 19 20 \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"