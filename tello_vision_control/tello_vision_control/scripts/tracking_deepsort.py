"""
This script is used to track objects with a neural network trained for objects detection (SSD Mobilenet) and
a tracker (DeepSort). Objects that can be tracked are objects in the coco.names file corresponding to the objects
for wich the SSD mobilenet network has been trained to detect.

"""

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from tello_vision_control import tools

########## SETUP ##########
###########################

##### Init detetction network (SSD Mobilenet)
# Init network
net = tools.initDetectionModel(weightsPath=None, configPath=None) # paths set to default
detection_threshold = 0.6

# Get objects labels
classNames = tools.getClassNames(fileAdress=None) # path set to default
trackedClass = ["person"]


##### Init tracker (DeepSort)
tracker = DeepSort(max_age=5)


##### Init video
video_adress = None
if video_adress is not None:
    # Video read from adress provided
    video = cv2.VideoCapture(video_adress)
else:
    # Video read from camera of the device
    video = cv2.VideoCapture(0)


########## LOOP ###########
###########################
while True:
    
    # Read video
    success, frame = video.read()
    if not success:
        break
    
    # Objects detection in frame
    classIds, confs, bbox = net.detect(frame, confThreshold=detection_threshold)
        
    # Creating lists of objects to be tracked and other objects in lists of 
    # ([left,top,w,h], confidence, detection_class_id)
    filteredDetections, otherDetections = tools.getDetectionsList(
        classIds, confs, bbox, classNames, filteredClass=trackedClass)
    
    print("filtered detections : " + str(filteredDetections))
    print("other detections : " + str(otherDetections))
    
    # Track and show objects to be tracked
    if filteredDetections:
        
        # Update tracker 
        tracks = tracker.update_tracks(filteredDetections, frame=frame) 

        # Track each detection
        for track in tracks:
            
            # Check if tracking is confirmed
            if track.is_confirmed():
                track_id = track.track_id
                track_box = [int(x) for x in track.to_ltrb()]
                track_object_id = track.det_class
                track_conf = track.get_det_conf()
                if track_conf != None:
                    track_conf = round(track.get_det_conf(), 3)
                      
                # Draw tracked objects in green         
                cv2.rectangle(frame, track_box, color=(0,255,0), thickness=2)
                cv2.putText(frame, tools.getClassNameFromId(classNames, track_object_id) + ", " + str(track_conf), (track_box[0]+10, track_box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                
            else:
                print("Passed track : " + str(track))
                
    # Show other objects
    if otherDetections:
        for detection in otherDetections:
            
            # Get object data
            detection_box = detection[0]
            detection_conf = detection[1]
            detection_class = detection[2]
            
            # Draw other objects in red
            cv2.rectangle(frame, detection_box, color=(0,0,255), thickness=2)
            cv2.putText(frame, tools.getClassNameFromId(classNames, detection_class) + ", " + str(detection_conf), 
                        (detection_box[0]+10, detection_box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    # Show image
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()