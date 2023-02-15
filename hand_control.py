import cv2
import mediapipe as mp
import sys
sys.path.append('handGestureModel')
from handGestureModel.model import KeyPointClassifier
import tools

# Init video
# video = cv2.VideoCapture(0)
video = cv2.VideoCapture("videos/two_hands_open_close.mp4")

w_image = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h_image = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Hand poses detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Hand poses classification
pose_classifier = KeyPointClassifier()
hand_states_labels = tools.get_labels_list('handGestureModel/model/keypoint_classifier/keypoint_classifier_label.csv')

##### Real-time tracking
while True:
    
    success, frame = video.read()
    if not success:
        break

    ### Hand poses detection
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_RGB)
       
    if results.multi_hand_landmarks:
        # For each landmarks group (hand)
        for hand_landmarks in results.multi_hand_landmarks:
            
            ### Hand pose classification 
            hand_bbox = tools.get_landmarks_box(hand_landmarks, w_image, h_image)
            landmark_coord_list = tools.get_landmark_coord_list(hand_landmarks, w_image, h_image)
            pre_processed_landmark_list = tools.pre_process_landmark(landmark_coord_list)
    
            hand_state_id = pose_classifier(pre_processed_landmark_list)
            hand_state = hand_states_labels[hand_state_id]
            
            # Drawings
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, hand_bbox, color=(0,255,0), thickness=2)
            cv2.putText(frame, str(hand_state), (hand_bbox[0]+10, hand_bbox[1]+30), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    
    # show the final output
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()