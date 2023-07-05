python scenes/scene_generate.py 1a 0 0 $1
python scenes/scene_generate.py 1a 0 1 $1
python scenes/scene_generate.py 1a 1 0 $1
python scenes/scene_generate.py 1a 1 1 $1
python scenes/scene_generate.py 1a 2 0 $1
python scenes/scene_generate.py 1a 2 1 $1
python scenes/scene_generate.py 4a 0 0 $1
python scenes/scene_generate.py 4a 0 1 $1
python scenes/scene_generate.py 4a 1 0 $1
python scenes/scene_generate.py 4a 1 1 $1
python scenes/scene_generate.py 4a 2 0 $1
python scenes/scene_generate.py 4a 2 1 $1
cp dataset/list.json dataset/$1/list.json
cp dataset/name_map.json dataset/$1/name_map.json
cp dataset/room_types.json dataset/$1/room_types.json
cp dataset/test_env.json dataset/$1/test_env.json
cp dataset/object_scale.json dataset/$1/object_scale.json
python scenes/scene_eval.py $1

ps ux | grep port\ 3072 | awk {'print $2'} | xargs kill

python3 tdw-gym/challenge.py \
--output_dir results \
--run_id 9999 \
--port 3072 \
--agents h_agent \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 10 \
--data_prefix dataset/$1/ \
--debug

rm -rf results/9999