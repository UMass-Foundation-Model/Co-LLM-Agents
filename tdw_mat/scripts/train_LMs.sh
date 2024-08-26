lm_id=gpt-4
port=30004
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id $lm_id \
--run_id train_LMs-$lm_id \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_train/ \
--eval_episodes 6 0 11 17 18 1 2 3 21 22 23 4 5 19 20 \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"
