from tello_vision_control import drone_tools
from tello_vision_control import vision_tools
import cv2

########## SETUP ##########
###########################

#### Cascade Classifier
classifier = vision_tools.createHaarCascadeClassifier()

#### Parameters
w = 360
h = 240
vidCenter = (int(w/2), int(h/2))
debug = False


#### Initialization
drone = drone_tools.initTello()

# PID relative to Yaw, y, z, in the drone frame
PID_YAW = drone_tools.PID(0.5, 0, 0)
PID_X = drone_tools.PID(3, 0, 0)
PID_Z = drone_tools.PID(0.5, 0, 0)


if not debug:
    print('Drone taking off')
    drone.takeoff()
    print('Takeoff done')


########## LOOP ###########
###########################
run = True
while run:
    
    # Exit if ENTER pressed
    k = cv2.waitKey(1) & 0xff
    if k == 13 :
        run = False
    
    frame = drone_tools.getTelloFrame(drone, w, h)
    
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(imgGray, 1.2, 4)
    success, frame, faceCoord = vision_tools.showAndFilterFaces(frame, faces)
    
    err = drone_tools.getError(success, faceCoord, vidCenter)
    
    drone_tools.controlDrone(drone, err, PID_Z, PID_YAW, PID_X=PID_X, debug=debug)
    
    cv2.imshow("Tello camera", frame)
    cv2.waitKey(1)
    
   
if not debug:
    print('Drone landing')
    drone.land()
    print('Landing done')