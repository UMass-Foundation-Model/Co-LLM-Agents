ps ux | grep port\ 2087 | awk {'print $2'} | xargs kill

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id gpt-4 \
--run_id 5001 \
--port 2087 \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--screen_size 256 \
--debug

ps ux | grep port\ 2087 | awk {'print $2'} | xargs kill