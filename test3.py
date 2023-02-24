import tello_vision_control
import cv2

video = cv2.VideoCapture("videos/two_hands_open_close.mp4")
tello_vision_control.tools.getVideoCenterCoord(video)
tello_vision_control.tello_tools.initTello()

# tello_vision_control.tello_tools.findFace()