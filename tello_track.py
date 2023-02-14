from tello_tools import *
import keyboard
from djitellopy import tello
import cv2
import matplotlib.pyplot as plt

#### Parameters
w = 360
h = 240
vidCenter = (int(w/2), int(h/2))
debug = False


#### Initialization
drone = initTello()

# PID relative to Yaw, y, z, in the drone frame
PID_YAW = PID(0.5, 0, 0.5)
PID_X = PID(3, 0, 3)
PID_Z = PID(0.5, 0, 0.5)

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
    
    if keyboard.is_pressed('l'):
        print('Drone landing')
        # tello.land(drone)
        print('Landing done')
        break
    
    frame = getTelloFrame(drone, w, h)
    success, frame, faceCoord = findFace(frame)
    err = getError(success, faceCoord, vidCenter)
    
    # print(f"faceCoord[0] = {str(faceCoord[0])}")
    # print(f"vidCenter[0] = {str(vidCenter[0])}")
    # print(f"err = {str(err)}")
    
    com_rc = trackFace(drone, PID_X, PID_Z, PID_YAW, err, debug=debug)
    
    # print(f"u_yaw = {str(com_rc)}")
    
    U.append(com_rc)
    Err.append(err)
    T.append(t)
    
    t += 1
    
    cv2.imshow("Tello camera", frame)
    cv2.waitKey(1)

# plt.plot(T, U, label = "Input")
# plt.plot(T, Err, label = "Error")
# plt.legend()
# plt.show()