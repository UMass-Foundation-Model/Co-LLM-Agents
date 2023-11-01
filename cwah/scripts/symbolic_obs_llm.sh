kill -9 $(lsof -t -i :6310)
python testing_agents/test_symbolic_LLM.py \
--mode single_LLM_gpt-4 \
--prompt_template_path LLM/prompt_single.csv \
--agent_num 1 \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6310 \
--lm_id gpt-4 \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--debug \