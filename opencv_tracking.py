import tools

import cv2
import numpy as np

##### Init video
video = cv2.VideoCapture("videos/occlusion.mp4")
# video = cv2.VideoCapture(0)

vidCenter = tools.getVideoCenterCoord(video)

##### First frame, select object
success, frame = video.read()

if not success:
    print('Video not read')
   
tracker = tools.trackerCreate('TLD')
bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox)

initWidth = bbox[2]

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
        cv2.putText(frame, "Failed to track object", (100,80), cv2.QT_FONT_NORMAL, 0.75,(0,0,255),2)

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