import cv2
from tello_vision_control import tools

##### Init video
video = cv2.VideoCapture("videos/xyz_slow.mp4")
# video = cv2.VideoCapture(0)

vidCenter = tools.getVideoCenterCoord(video)

##### Real-time tracking
while True:

    success, frame = video.read()
    if not success:
        break
        
    binImg= tools.binFilter(frame, 200)
    center = tools.getShapeCenter(binImg)
    
    cv2.circle(frame, center, 20, (0, 0, 255), -1)
    cv2.arrowedLine(frame, vidCenter, center, (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, f"Object coordinates : ({str(center[0])}, {str(center[1])})" ,
                    (100,80), cv2.QT_FONT_NORMAL, 0.75,(255,0,0),2)

    cv2.imshow("Tracking", frame)   

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()