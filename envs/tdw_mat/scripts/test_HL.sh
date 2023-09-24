port=10003
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--experiment_name HL \
--run_id HL_1 \
--port $port \
--agents h_agent lm_agent \
--prompt_template_path LLM/prompt_com.csv \
--communication \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 11 17 18 1 2 3 21 22 23 4 5 6 7 8 9 10 12 13 14 15 16 19 20 \
--debug \
--screen_size 256 \
--no_save_img

pkill -f -9 "port $port"