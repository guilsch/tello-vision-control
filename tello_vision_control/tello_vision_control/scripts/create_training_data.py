#!/usr/bin/env python

"""
This script can be used to create training data from the position of the hand detected with the camera.

- classifier_labels_files must be a string containing the adress of the csv file with the different labels (eg. 
  "Open", "Close"), each line number corresponding to the label id.
- keypoints_output_cvs must be an string adress of a csv file where each data will be stored.
- video_adress is for the video that will be read, if None, camera of the device will be used.

During the video being read, press a number from 0 to 9 corresponding to the label id (eg. 0 for open, 1 for close)
when a hand position is detected, the hand pose will be add to the csv datafile. Then, press "p" to continue the 
video.

"""

import cv2
import mediapipe as mp
from tello_vision_control import classification_utils
from tello_vision_control import tools

##### Parameters
classifier_labels_file = None
keypoints_output_csv = 'keypoint.csv'
video_adress = None

##### Initialization
# Init video
if video_adress is not None:
    # Video read from adress provided
    video = cv2.VideoCapture(video_adress)
else:
    # Video read from camera of the device
    video = cv2.VideoCapture(0)

w_image = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h_image = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Hand poses detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Hand poses classification
hand_states_labels = tools.get_labels_list(classifier_labels_file)

##### Real-time tracking
pause = False
while True:
    
    # Read video
    success, frame = video.read()
    if not success:
        break

    # Read keyboard
    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break

    ### Hand poses detection
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_RGB) 
    
    ### For all the poses detected
    if results.multi_hand_landmarks:
        # For each landmarks group (hand)
        for hand_landmarks in results.multi_hand_landmarks:
            
            ### Hand pose pre-processing
            hand_bbox = tools.get_landmarks_box(hand_landmarks, w_image, h_image)
            landmark_coord_list = tools.get_landmark_coord_list(hand_landmarks, w_image, h_image)
            pre_processed_landmark_list = tools.pre_process_landmark(landmark_coord_list)
    
            # Read number pressed by user
            label_id = classification_utils.get_label_id_from_keyboard(key)
            
            # Add pose to database
            if label_id is not None and label_id < len(hand_states_labels):
                pause = True
                print(label_id)
                classification_utils.write_new_data(label_id, pre_processed_landmark_list,
                                                    csv_path=keypoints_output_csv)
                hand_state = hand_states_labels[label_id]
                
                cv2.putText(frame, hand_state, (10, 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)               
            
            # Drawings
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, hand_bbox, color=(0,255,0), thickness=2)
    
    # show the final output
    cv2.imshow("Camera", frame)

    # Manage pause
    if pause:
        while True:
            key = cv2.waitKey(10)
            if key == ord("p"):
                pause = False
                break

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()