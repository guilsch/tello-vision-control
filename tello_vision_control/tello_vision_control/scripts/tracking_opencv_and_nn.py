import cv2
from tello_vision_control import tools

########## SETUP ##########
###########################

##### Init tracker  
trackerType = 'MEDIANFLOW'
tracker = tools.trackerCreate(trackerType)


##### Init detetction network (SSD Mobilenet) in case tracking is lost
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

######## PRE-LOOP #########
###########################

# First frame, select object
success, frame = video.read()

# Init tracker
bboxInit = cv2.selectROI(frame, False)
tracker.init(frame, bboxInit)

# Define depth of reference
initWidth = bboxInit[2]


########## LOOP ###########
###########################
while True:

    # Read video 
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
        #### If tracker lost the object, try to detect it with NN and redefine tracker with the new detection
        print("Failed to track object")
        cv2.putText(frame, "Failed to track object", (100,80), cv2.QT_FONT_NORMAL, 0.75,(0,0,255),2)
         
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
        
        # Draw best detection and re-initialize tracker
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
            
            # Re-initialize tracker
            print("Re-initialization of tracker")
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