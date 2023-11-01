import shutil
import sys
import os
import subprocess
import numpy as np
import cv2
import argparse
def create_video(video_folder, format, out_file):
    file_and_format = f'{video_folder}/{format}'
    out_dir = os.path.join('Output', out_file)
    # frames = 10
    # for i in range(1, frames):
    #     print(file_and_format)
    #     frame_num = file_and_format.replace('*', '{:03}'.format(i))
    #     prev_frame_num = file_and_format.replace('*', '{:03}'.format(i-1))
    #     print(prev_frame_num)
    #     if not os.path.isfile(frame_num):
    #         shutil.copy(prev_frame_num, frame_num)

    frame_rate = 2
    midname = 'demowww.mp4'
    subprocess.call(['/work/pi_chuangg_umass_edu/shanjiaming/watch_and_help/ffmpeg-6.0-amd64-static/ffmpeg',
                    '-framerate', str(frame_rate),
                    '-i',
                     '{}/{}'.format(video_folder, format.replace('*', '%03d')),
                     '-pix_fmt', 'yuv420p',
                     midname])
    # os.system(f'ffmpeg -i {midname} -vf "transpose=2" {out_dir}.mp4')
    # os.system(f'rm {midname}')


# def merge_frames(in_formats, nframes, out_format):
#     for i in range(0, nframes):
#         curr_imgs = []
#         for in_format in in_formats:

#             frame_num = in_format.replace('*', str(i))
#             prev_frame_num = in_format.replace('*', (i - 1))

#             if not os.path.isfile(frame_num):
#                 shutil.copy(prev_frame_num, frame_num)
#             curr_imgs.append(cv2.imread(frame_num))
#         img_join = np.concatenate(curr_imgs, 1)
#         cv2.imwrite(out_format.replace('*', str(i)), img_join)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='/work/pi_chuangg_umass_edu/icefox/test_results/vision_ML_chatgpt_cot_record_2/image/Bob', type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    args = parser.parse_args()
    # merge_frames(['/Users/xavierpuig/Desktop/test_videos/bob_with_info/action_*.png',
    #               '/Users/xavierpuig/Desktop/test_videos/alice_with_info/action_*.png'], 570,
    #              '/Users/xavierpuig/Desktop/test_videos/merged2_info/Action_*_normal.png')
    # create_video('/Users/xavierpuig/Desktop/test_videos/merged2_info/',y
    #              'Action_*_normal.png',
    #              '/Users/xavierpuig/Desktop/test_videos/alice_and_bob_info.mp4')
    create_video(args.dir,
                 '*_img.png',
                 args.output_dir)
    #create_video(sys.argv[1], sys.argv[2], sys.argv[3])