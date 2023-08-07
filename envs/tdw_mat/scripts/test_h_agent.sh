ps ux | grep port\ 3062 | awk {'print $2'} | xargs -9 kill

python3 tdw-gym/challenge.py \
--output_dir results \
--run_id 2001 \
--port 3062 \
--agents h_agent h_agent \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/dataset_test/ \
--debug

ps ux | grep port\ 3062 | awk {'print $2'} | xargs -9 kill