python scenes/scene_generate.py 2a 0 0 $1
python scenes/scene_generate.py 2a 0 1 $1
python scenes/scene_generate.py 2a 1 0 $1
python scenes/scene_generate.py 2a 1 1 $1
python scenes/scene_generate.py 2a 2 0 $1
python scenes/scene_generate.py 2a 2 1 $1
python scenes/scene_generate.py 5a 0 0 $1
python scenes/scene_generate.py 5a 0 1 $1
python scenes/scene_generate.py 5a 1 0 $1
python scenes/scene_generate.py 5a 1 1 $1
python scenes/scene_generate.py 5a 2 0 $1
python scenes/scene_generate.py 5a 2 1 $1
cp dataset/list.json dataset/$1/list.json
cp dataset/name_map.json dataset/$1/name_map.json
cp dataset/room_types.json dataset/$1/room_types.json
cp dataset/test_env.json dataset/$1/test_env.json
cp dataset/object_scale.json dataset/$1/object_scale.json
python scenes/scene_eval.py $1