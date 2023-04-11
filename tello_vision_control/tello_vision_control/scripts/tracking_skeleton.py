""" This script shows a skeleton of persons in the image and compute coordinate of a particluar part of the body
    (default is nose)
"""

import cv2
import mediapipe as mp
from tello_vision_control import tools
from tello_vision_control import tello_tools


########## SETUP ##########
###########################

##### Init video
video_adress = None
if video_adress is not None:
    # Video read from adress provided
    video = cv2.VideoCapture(video_adress)
else:
    # Video read from camera of the device
    video = cv2.VideoCapture(0)

w_image = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h_image = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

##### initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Select the pose you want to track
# Check the documentation :
# https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
landmark_num = 0 


########## LOOP ###########
###########################
while True:
    
    success, frame = video.read()
    if not success:
        break

    # Detect poses
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = pose_detector.process(frame_RGB)

    # Get coordinates
    success, landmark_coord3D = tools.getLandmarkCoord(detection, landmark_num, w_image, h_image)
    
    # draw detected skeleton on the frame
    mp_drawing.draw_landmarks(frame, detection.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if success:
        # print("Depth (m) : " + str(landmark_coord3D[2]))
        cv2.circle(frame, (landmark_coord3D[0], landmark_coord3D[1]), 10, (0, 255, 0), -1)
        
    # show the final output
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()