""" This script shows a skeleton of persons in the image and compute coordinate of a particluar part of the body
    (default is nose). The drone tracks it.
"""

import cv2
import mediapipe as mp
from tello_vision_control import tools
from tello_vision_control import tello_tools

########## SETUP ##########
###########################

##### Init video
w_image = 360
h_image = 240
vidCenter = (int(w_image/2), int(h_image/2))

##### initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# PID relative to Yaw, X, Z, in the drone frame
PID_YAW = tello_tools.PID(0.5, 0, 0.5)
PID_X = tello_tools.PID(0.5, 0, 0.5)
PID_Z = tello_tools.PID(0.5, 0, 0.5)

PID_YAW_Manager = tello_tools.PIDManager(PID_YAW)
PID_X_Manager = tello_tools.PIDManager(PID_X)
PID_Z_Manager = tello_tools.PIDManager(PID_Z)

##### Param
# Select the landmark you want to track. Check the documentation :
# https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
landmark_num = 0 
dist = 100      # Distance reference from the drone in centimeters
debug = False   # Mode


######## PRE-LOOP #########
###########################

#### Tello initialization
drone = tello_tools.initTello()

if not debug:
    print('Drone taking off')
    drone.takeoff()
    print('Takeoff done')

########## LOOP ###########
###########################
run = True
while run:
    
    # Exit if ENTER pressed
    k = cv2.waitKey(1) & 0xff
    if k == 13 :
        run = False
    
    # Read frame
    frame = tello_tools.getTelloFrame(drone, w_image, h_image)

    # Detect poses
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = pose_detector.process(frame_RGB)

    # Get coordinates
    success, landmark_coord3D = tools.getLandmarkCoord(detection, landmark_num, w_image, h_image)
    
    # Track
    err = tello_tools.getError(success, landmark_coord3D, vidCenter, err_X_ref=dist, area_mode=False, mode_3D=True)
    com_rc = tello_tools.controlDrone(drone, err, PID_Z, PID_YAW, PID_X=PID_X, debug=debug)
    
    print("com_rc = " + str(com_rc) + "     ; distance = " + str(landmark_coord3D[2]))
    
    # draw detected skeleton on the frame
    mp_drawing.draw_landmarks(frame, detection.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if success:
        cv2.circle(frame, (landmark_coord3D[0], landmark_coord3D[1]), 10, (0, 255, 0), -1)
        
    # show the final output
    cv2.imshow("Tracking", frame)

    # Update PID gains
    PID_YAW_Manager.update()
    PID_X_Manager.update()
    PID_Z_Manager.update()
    
if not debug:
    print('Drone landing')  
    drone.land()
    print('Landing done')
