kill -9 $(lsof -t -i :6415)
python testing_agents/test_vision_LLM.py \
--mode vision_ML_comm_gpt-4 \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--opponent-subgoal comm \
--satisfied-comm \
--obs_type normal_image \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6415 \
--use-alice \
--lm_id gpt-4 \
--source openai \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--debug