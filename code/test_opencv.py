import os, sys
import cv2

video_path = "/home/yuanma/Datasets/SmartEye_D0/2017_11_10 huangxiaolin.mp4"
video = cv2.VideoCapture(video_path)

while True:
    ret, frame = video.read()

    if ret is False:
        break

    cv2.imshow("Video", frame)
    cv2.waitKey(0)

video.release()
