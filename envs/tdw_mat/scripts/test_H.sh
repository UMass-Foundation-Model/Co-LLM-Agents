port=10001
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--experiment_name H \
--run_id run_2 \
--port $port \
--agents h_agent \
--max_frames 3000 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 11 17 18 1 2 3 21 22 23 4 5 6 7 8 9 10 12 13 14 15 16 19 20 \
--screen_size 256 \
--no_save_img \
--debug

pkill -f -9 "port $port"
