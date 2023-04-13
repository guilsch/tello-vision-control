#!/usr/bin/env python

"""
This script can be used to detect hand poses and control the tello drone from them.

- classifier_file is a string adress of the tflite trained model used for classification
- classifier_labels_files must be a string containing the adress of the csv file with the different labels (eg. 
  "Open", "Close"), each line number corresponding to the label id.
- video_adress is for the video that will be read, if None, camera of the device will be used.
- drone_mode must be True is tello drone is connected, otherwise hand pose detection will occur on the provided 
  video
"""

import cv2
import mediapipe as mp
from tello_vision_control import vision_tools
from tello_vision_control import classification_tools
from tello_vision_control import drone_tools


########## SETUP ##########
###########################

##### Parameters
classifier_file = None
classifier_labels_file = None
video_adress = None
drone_mode = False

##### Initialization
# Hand poses detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Hand poses classification
pose_classifier = classification_tools.KeyPointClassifierLoader(classifier_file)
hand_states_labels = vision_tools.get_labels_list(classifier_labels_file)

# Drone Initialization
if drone_mode:
    drone = drone_tools.initTello()

# Init video
if drone_mode:
    # Video read from drone
    video = drone_tools.getTelloFrame(drone, 360, 240)   
elif video_adress is not None:
    # Video read from adress provided
    video = cv2.VideoCapture(video_adress)
else:
    # Video read from camera of the device
    video = cv2.VideoCapture(0)

w_image = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h_image = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

########## LOOP ###########
###########################
if drone_mode:
    drone.takeoff()
    print('Takeoff done')

while True:
    
    ### Read video
    if drone_mode:
        frame = drone_tools.getTelloFrame(drone, w_image, h_image)
    else:     
        success, frame = video.read()
        if not success:
            break

    ### Hand poses detection
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_RGB)
       
    hand_state_id = 0
    ### For all the poses detected   
    if results.multi_hand_landmarks:
        # For each landmarks group (hand)
        for hand_landmarks in results.multi_hand_landmarks:
            
            ### Hand pose classification 
            hand_bbox = vision_tools.get_landmarks_box(hand_landmarks, w_image, h_image)
            landmark_coord_list = vision_tools.get_landmark_coord_list(hand_landmarks, w_image, h_image)
            pre_processed_landmark_list = vision_tools.pre_process_landmark(landmark_coord_list)
    
            hand_state_id = pose_classifier(pre_processed_landmark_list)
            hand_state = hand_states_labels[hand_state_id]
            
            # Drawings
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, hand_bbox, color=(0,255,0), thickness=2)
            cv2.putText(frame, str(hand_state), (hand_bbox[0]+10, hand_bbox[1]+30), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    
            # Trickshot if particular hand pose has been detected
            if drone_mode:
                drone_tools.do_trickshot_from_hand_position_id(drone, hand_state_id)
    
    # show the final output
    cv2.imshow("Camera", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        if drone_mode:
            drone.land()        
        break

if not drone_mode:
    video.release()