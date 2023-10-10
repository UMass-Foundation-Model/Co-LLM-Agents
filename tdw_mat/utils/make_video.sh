result_dir=results/Llama-2-13b-hf/0

ffmpeg -i ${result_dir}/top_down_image/img_%05d.jpg -pix_fmt yuv420p ${result_dir}/top_down_image.mp4

#ffmpeg -pattern_type glob -i "${result_dir}/Images/0/*_%d.png" -pix_fmt yuv420p ${result_dir}/image_0.mp4