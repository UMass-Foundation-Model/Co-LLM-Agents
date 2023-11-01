import cv2
import os
import argparse

def images_to_video(image_folder, video_name, fps):
    r'''
        support both first view and top-down view
    '''
    images = [img for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg")) and not img.endswith("seg.png") and not img.endswith("depth.png") and not img.endswith("map.png") and not img.endswith("filter.png")]

    if images[0].endswith(".png"):
        images.sort(key = lambda x: int(x.split('.')[0]))
    else:
        images.sort(key = lambda x: int(x.split('.')[0].split('_')[1]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="image path")
    parser.add_argument("--video_name", type=str, default="video.avi")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    images_to_video(args.image_folder, args.video_name, args.fps)