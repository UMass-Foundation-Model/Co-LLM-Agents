ps ux | grep port\ 3085 | awk {'print $2'} | xargs kill

python3 tdw-gym/challenge.py \
--output_dir results \
--run_id 3001 \
--port 3085 \
--agents h_agent lm_agent \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--data_prefix dataset/dataset_test/ \
--debug

ps ux | grep port\ 3085 | awk {'print $2'} | xargs kill