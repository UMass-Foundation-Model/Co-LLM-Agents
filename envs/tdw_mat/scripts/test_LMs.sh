lm_id=gpt-4
port=2087
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id $lm_id \
--run_id LMs-$lm_id \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--eval_episodes 11 17 18 \
--screen_size 256 \
--no_save_img \
--debug

pkill -f -9 "port $port"