import tools

import cv2
import numpy as np

##### Init video
# video = cv2.VideoCapture("videos/xyz_slow.mp4")
video = cv2.VideoCapture(0)

vidCenter = tools.getVideoCenterCoord(video)

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

threshold = 50
trackedClass = "person"

##### First frame, select object
success, frame = video.read()

if not success:
    print('Video not read')

trackerType = 'MEDIANFLOW'   
tracker = tools.trackerCreate(trackerType)
bboxInit = cv2.selectROI(frame, False)
ok = tracker.init(frame, bboxInit)

initWidth = bboxInit[2]

##### Real-time tracking
while True:

    success, frame = video.read()
    if not success:
        break
        
    # Start timer
    start = cv2.getTickCount()

    # Update tracker
    success, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start)

    # Draw bounding box
    if success:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                
        bboxCoord = tools.getBoxCenterCoord3D(bbox, initWidth)
        cv2.putText(frame, f"Object coordinates : ({str(bboxCoord[0])}, {str(bboxCoord[1])}, {str(bboxCoord[2])})" ,
                    (100,80), cv2.QT_FONT_NORMAL, 0.75,(255,0,0),2)
        
        cv2.circle(frame, bboxCoord[:2], 2, (0, 0, 255), -1)
        cv2.arrowedLine(frame, vidCenter, bboxCoord[:2], tools.createColorGrading(bboxCoord[2]), 2, tipLength=0.3)
    
    else :
        # If tracker lost the object, try to detect it with NN and redefine tracker with the new detection
        cv2.putText(frame, "Failed to track object", (100,80), cv2.QT_FONT_NORMAL, 0.75,(0,0,255),2)
         
        classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
        if len(classIds) != 0:
        
            maxConfidence = 0
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):                    
                if confidence * 100 > threshold:
                    if classNames[classId-1] == trackedClass:
                        if confidence >= maxConfidence:
                            
                            maxConfidence = confidence
                            boxCoord = tools.getBoxCenterCoord2D(box)
                            
                            cv2.rectangle(frame, box, color=(0,255,0), thickness=2)
                            cv2.putText(frame, classNames[classId-1].upper(),(box[0]+10, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                            cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                            
                            cv2.circle(frame, boxCoord, 2, (0, 0, 255), -1)
                            cv2.arrowedLine(frame, vidCenter, boxCoord,(0, 0, 255), 2, tipLength=0.3)
                            
                            tracker = tools.trackerCreate(trackerType)
                            tracker.init(frame, tuple(box))
        

    # Draw center
    cv2.circle(frame, vidCenter, 2, (0, 0, 255), -1)

    # Display
    cv2.putText(frame, " Tracker", (100,20), cv2.QT_FONT_NORMAL, 0.75, (50,170,50),2)
    cv2.putText(frame, f"FPS : {str(int(fps))}", (100,50), cv2.QT_FONT_NORMAL, 0.75, (50,170,50), 2)
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()