ps ux | grep port\ 1087 | awk {'print $2'} | xargs -9 kill

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id gpt-4 \
--run_id 4005 \
--port 1087 \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--debug

ps ux | grep port\ 1087 | awk {'print $2'} | xargs -9 kill