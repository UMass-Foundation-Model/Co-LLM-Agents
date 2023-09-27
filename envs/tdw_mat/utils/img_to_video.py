import cv2
import os

def images_to_video(image_folder, video_name, fps): # img.endswith(".png") and
    images = [img for img in os.listdir(image_folder) if not img.endswith("seg.png") and not img.endswith("depth.png") and not img.endswith("map.png") and not img.endswith("filter.png")]
    images.sort(key = lambda x: int(x.split('.')[0].split('_')[1]))
    print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

image_folder = 'results/vision-LMs-Llama-2-13b-chat-hf/run_1/2/top_down_image'  #img path
video_name = 'video_top.avi'
fps = 15

images_to_video(image_folder, video_name, fps)