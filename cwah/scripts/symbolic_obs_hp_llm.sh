kill -9 $(lsof -t -i :6315)
python testing_agents/test_symbolic_LLM.py \
--mode ML_comm_gpt-4 \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6315 \
--use-alice \
--opponent-subgoal comm --satisfied-comm \
--lm_id gpt-4 \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--debug