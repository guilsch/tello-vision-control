from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import tools

##### Init network

configPath = "ssd_mobilenet_v3_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "ssd_mobilenet_v3_files/frozen_inference_graph.pb"
classFile = "ssd_mobilenet_v3_files/coco.names"
  
classNames = tools.getClassNames(classFile)

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


##### 
tracker = DeepSort(max_age=5)

#####

video = cv2.VideoCapture("videos/xyz_slow.mp4")
# video = cv2.VideoCapture(0)
trackedClass = ["apple"]
detection_threshold = 0.6

# bbs = object_detector.detect(frame)
while True:
    
    success, frame = video.read()
    if not success:
        break
    
    # Objects detection in frame
    classIds, confs, bbox = net.detect(frame, confThreshold=detection_threshold)
        
    # Creating list of detections
    filteredDetections, otherDetections = tools.getDetectionsList(classIds, confs, bbox, classNames, filteredClass=trackedClass)
    
    print("filtered detections : " + str(filteredDetections))
    print("other detections : " + str(otherDetections))
    
    # Track and show filtered objects
    if filteredDetections:
        # Update tracker
        # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        tracks = tracker.update_tracks(filteredDetections, frame=frame) 

        # Track each detection
        for track in tracks:
            
            # Check tracking
            if track.is_confirmed():
                track_id = track.track_id
                track_box = [int(x) for x in track.to_ltrb()]
                track_object_id = track.det_class
                track_conf = track.get_det_conf()
                if track_conf != None:
                    track_conf = round(track.get_det_conf(), 3)
                                
                cv2.rectangle(frame, track_box, color=(0,255,0), thickness=2)
                cv2.putText(frame, tools.getClassNameFromId(classNames, track_object_id) + ", " + str(track_conf), (track_box[0]+10, track_box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                
            else:
                print("passed " + str(track))
                
    # Show other objects
    if otherDetections:
        for detection in otherDetections:
            detection_box = detection[0]
            detection_conf = detection[1]
            detection_class = detection[2]
            cv2.rectangle(frame, detection_box, color=(0,0,255), thickness=2)
            cv2.putText(frame, tools.getClassNameFromId(classNames, detection_class) + ", " + str(detection_conf), 
                        (detection_box[0]+10, detection_box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break