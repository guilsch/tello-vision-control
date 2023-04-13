""" This script shows a skeleton of persons in the image and compute coordinate of a particluar part of the body
    (default is nose). The drone tracks it along Z axis and Yaw.
"""

import cv2
import mediapipe as mp
from tello_vision_control import vision_tools
from tello_vision_control import drone_tools
from tello_vision_control import classification_tools

########## SETUP ##########
###########################

##### Init video
w_image = 360
h_image = 240
vidCenter = (int(w_image/2), int(h_image/2))

##### initialize body pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

##### initialize hand pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Hand poses classification
tflite_save_path = None
labels_path = None
pose_classifier = classification_tools.KeyPointClassifierLoader(tflite_save_path)
hand_states_labels = vision_tools.get_labels_list(labels_path)

# PID relative to Yaw, X, Z, in the drone frame
PID_YAW = drone_tools.PID(0.5, 0, 0.5)
PID_Z = drone_tools.PID(0.5, 0, 0.5)

PID_YAW_Manager = drone_tools.PIDManager(PID_YAW)
PID_Z_Manager = drone_tools.PIDManager(PID_Z)

##### Param
# Select the landmark you want to track. Check the documentation :
# https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
landmark_num = 0
debug = False   # Mode


######## PRE-LOOP #########
###########################

#### Tello initialization
drone = drone_tools.initTello()

if not debug:
    print('Drone taking off')
    drone.takeoff()
    print('Takeoff done')

########## LOOP ###########
###########################
run  = True
while run:
    
    k = cv2.waitKey(1) & 0xff
    # Exit if ENTER pressed
    if k == 13 :
        run = False
    # Stop all motors if SPACE pressed
    if k == 32:
        run = False
        drone.emergency()
    
    # Read frame
    frame = drone_tools.getTelloFrame(drone, w_image, h_image)

    # Detect body and hand poses
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    body_detection = pose_detector.process(frame_RGB)
    hand_detection = hands.process(frame_RGB)

    ##### Tracking
    # Get body part coordinates
    success, landmark_coord3D = vision_tools.getLandmarkCoord(body_detection, landmark_num, w_image, h_image)
    
    # Track    
    err = drone_tools.getError(success, landmark_coord3D, vidCenter, area_mode=False, mode_3D=False)
    com_rc = drone_tools.controlDrone(drone, err, PID_Z, PID_YAW, PID_X=None, debug=debug)
        
    # draw detected skeleton on the frame
    mp_drawing.draw_landmarks(frame, body_detection.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if success:
        cv2.circle(frame, (landmark_coord3D[0], landmark_coord3D[1]), 10, (0, 255, 0), -1)
        
    # Update PID gains
    # PID_YAW_Manager.update()
    # PID_X_Manager.update()
    # PID_Z_Manager.update()
    ##### End tracking
        
    ##### Hand control
    hand_state_id = 0
    ### For all the poses detected   
    if hand_detection.multi_hand_landmarks:
        # For each landmarks group (hand)
        for hand_landmarks in hand_detection.multi_hand_landmarks:
            
            ### Hand pose classification 
            hand_bbox = vision_tools.get_landmarks_box(hand_landmarks, w_image, h_image)
            landmark_coord_list = vision_tools.get_landmark_coord_list(hand_landmarks, w_image, h_image)
            pre_processed_landmark_list = vision_tools.pre_process_landmark(landmark_coord_list)
    
            hand_state_id = pose_classifier(pre_processed_landmark_list)
            hand_state = hand_states_labels[hand_state_id]
            print("Hand position detected : " + str(hand_state))
            
            # Drawings
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, hand_bbox, color=(0,255,0), thickness=2)
            cv2.putText(frame, str(hand_state), (hand_bbox[0]+10, hand_bbox[1]+30), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            # Control
            if hand_state_id==2:
                print("Landing")
                if not debug:
                    drone.land()
                run = False
    ##### End hand control
        
    print("com_rc = " + str(com_rc) + "     ; distance = " + str(landmark_coord3D[2]))
        
    # show the final output
    cv2.imshow("Tracking", frame)
    
print('Drone landing')
drone.land()
print('Landing done')