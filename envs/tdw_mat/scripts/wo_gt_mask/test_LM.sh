ps ux | grep port\ 1091 | awk {'print $2'} | xargs kill

python3 tdw-gym/challenge.py \
--output_dir results \
--run_id 100010 \
--port 1091 \
--agents lm_agent \
--lm_id gpt-4 \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--debug

ps ux | grep port\ 1091 | awk {'print $2'} | xargs kill