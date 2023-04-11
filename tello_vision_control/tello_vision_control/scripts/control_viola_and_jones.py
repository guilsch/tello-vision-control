from tello_vision_control.utils import tello_tools
import keyboard
import cv2


#### Parameters
w = 360
h = 240
vidCenter = (int(w/2), int(h/2))
debug = False


#### Initialization
drone = tello_tools.initTello()

# PID relative to Yaw, y, z, in the drone frame
PID_YAW = tello_tools.PID(0.5, 0, 0.5)
PID_X = tello_tools.PID(3, 0, 3)
PID_Z = tello_tools.PID(0.5, 0, 0.5)

PID_YAW_Manager = tello_tools.PIDManager(PID_YAW)
PID_X_Manager = tello_tools.PIDManager(PID_X)
PID_Z_Manager = tello_tools.PIDManager(PID_Z)


if not debug:
    print('Drone taking off')
    drone.takeoff()
    print('Takeoff done')


#### Debug
U = []
Err = []
T = []
t = 0

#### Real-time
while True:
    
    if keyboard.is_pressed('l') and not debug:
        print('Drone landing')
        drone.land(drone)
        print('Landing done')
        break
    
    frame = tello_tools.getTelloFrame(drone, w, h)
    success, frame, faceCoord = tello_tools.findFace(frame)
    err = tello_tools.getError(success, faceCoord, vidCenter)
    
    com_rc = tello_tools.controlDrone(drone, PID_X, PID_Z, PID_YAW, err, debug=debug)
    
    U.append(com_rc)
    Err.append(err)
    T.append(t)
    
    t += 1
    
    cv2.imshow("Tello camera", frame)
    cv2.waitKey(1)
    
    PID_YAW_Manager.update()
    PID_X_Manager.update()
    PID_Z_Manager.update()