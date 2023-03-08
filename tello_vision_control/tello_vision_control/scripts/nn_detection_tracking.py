"""
This script is used to track objects with only a neural network trained for objects detection (SSD Mobilenet).
Objects that can be tracked are objects in the coco.names file corresponding to the objects for wich the SSD
mobilenet network has been trained to detect.

"""

import cv2
from tello_vision_control import tools

########## SETUP ##########
###########################

##### Init detetction network (SSD Mobilenet)
# Init network
net = tools.initDetectionModel(weightsPath=None, configPath=None) # paths set to default
detection_threshold = 0.6

# Get objects labels
classNames = tools.getClassNames(fileAdress=None) # path set to default
trackedClass = "person"


##### Init video
video_adress = None
if video_adress is not None:
    # Video read from adress provided
    video = cv2.VideoCapture(video_adress)
else:
    # Video read from camera of the device
    video = cv2.VideoCapture(0)

vidCenter = tools.getVideoCenterCoord(video)


########## LOOP ###########
###########################
while True:

    # Read video    
    success, frame = video.read()
    if not success:
        break

    # Objects detection in frame
    classIds, confs, bbox = net.detect(frame, confThreshold=detection_threshold)
    
    # For each detection in the frame, we take the one with the most confidence
    bestClassId, bestConfidence, bestBox, bestBoxCoord = None, None, None, None
    if len(classIds) != 0:
        
        maxConfidence = 0
        
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId-1] == trackedClass:
                if confidence >= maxConfidence:
                    # Save best detection
                    maxConfidence = confidence
                    bestBoxCoord = tools.getBoxCenterCoord2D(box)
                    bestClassId, bestConfidence, bestBox = classId, confidence, box
     
            else:
                # Draw other detections
                cv2.rectangle(frame, box, color=(0,0,255), thickness=1)
                cv2.putText(frame, classNames[classId-1].upper(),(box[0]+10, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    # Draw best detection
    if bestClassId is not None:
        # Draw rectangle and info
        cv2.rectangle(frame, bestBox, color=(0,255,0), thickness=2)
        cv2.putText(frame, classNames[bestClassId-1].upper(),(bestBox[0]+10, bestBox[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(frame,str(round(bestConfidence*100,2)),(bestBox[0]+200,bestBox[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
        # Draw arrow
        cv2.circle(frame, bestBoxCoord, 2, (0, 0, 255), -1)
        cv2.arrowedLine(frame, vidCenter, bestBoxCoord, (0, 0, 255), 2, tipLength=0.3)
    
    # Show frame
    cv2.circle(frame, vidCenter, 2, (0, 0, 255), -1)        
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break