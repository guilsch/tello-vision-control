import cv2

import tools

##### Init video
video = cv2.VideoCapture("videos/xyz_slow.mp4")
# video = cv2.VideoCapture(0)
vidCenter = tools.getVideoCenterCoord(video)

##### Init network

configPath = "ssd_mobilenet_v3_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "ssd_mobilenet_v3_files/frozen_inference_graph.pb"
classFile = "ssd_mobilenet_v3_files/coco.names"

# configPath = "yolo_v3_files/yolov3.cfg"
# weightsPath = "yolo_v3_files/yolov3.weights"
# classFile = "yolo_v3_files/coco.names"

classNames= []

with open(classFile, "r") as f:
    classNames = [ligne.strip() for ligne in f]

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    

    
##### Parameters
threshold = 50
trackedClass = "apple"

##### Real-time tracking
while True:

    success, frame = video.read()
    if not success:
        break

    # frame = tools.blurBG(frame)

    # Detect    
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    # classIds, confs, bbox = tools.detectWithYOLO(net, frame, 0.5)
    
    if len(classIds) != 0:
        
        maxConfidence = 0
        
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        # for classId, confidence, box in zip(classIds, confs, bbox):
                
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
    
                        
                else:   
                    cv2.rectangle(frame, box, color=(0,0,255), thickness=1)
                    cv2.putText(frame, classNames[classId-1].upper(),(box[0]+10, box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                    cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    # Show frame
    cv2.circle(frame, vidCenter, 2, (0, 0, 255), -1)        
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break