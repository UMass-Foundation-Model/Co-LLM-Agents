port=3068
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--experiment_name vision-single \
--run_id run_0 \
--port $port \
--agents h_agent \
--max_frames 3000 \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 11 17 18 1 2 3 21 22 23 4 5 6 7 8 9 10 12 13 14 15 16 19 20 \
--debug \
--screen_size 512 \
--no_gt_mask \
--no_save_img

pkill -f -9 "port $port"