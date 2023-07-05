kill -9 $(lsof -t -i :6386)
python testing_agents/test_symbolic_hp.py \
--mode symbolic_hp \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6386 \