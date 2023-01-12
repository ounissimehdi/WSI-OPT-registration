import os
import cv2

# image path
im_dir = 'images'
# output video path
video_dir = 'video'
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
# set saved fps
fps = 6
# get frames list
os.listdir(im_dir)
frames = sorted(os.listdir(im_dir), key = lambda x: int(x[:-4].split('_')[-1]))
# w,h of image
img = cv2.imread(os.path.join(im_dir, frames[0]))
img_size = (img.shape[1], img.shape[0])
# get seq name
seq_name = os.path.dirname(im_dir).split('/')[-1]
# splice video_dir
video_dir = os.path.join(video_dir, "CD31" + '.avi')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# if want to write .mp4 file, use 'MP4V'
videowriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for frame in frames:
    f_path = os.path.join(im_dir, frame)
    image = cv2.imread(f_path)
    videowriter.write(image)
    print(frame + " has been written!")
    print(image.shape)

videowriter.release()
