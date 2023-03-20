from djitellopy import Tello
import cv2
from tello_vision_control.utils import tools
import time
import numpy as np
import math


def initTello():
    
    drone = Tello()
    
    drone.connect()
    drone.streamoff()
    drone.streamon()
    
    print(f"Battery : {str(drone.get_battery())}%")
    print("Tello ready")
    
    return drone

def getTelloFrame(tello, w = 360, h = 240):
    
    frame = tello.get_frame_read()
    frame = frame.frame
    frame = cv2.resize(frame, (w, h))
    
    return frame

def findFace(img):
    """
    Return the image with a rectangle where a face has been detected, 
    as well as the (x, y) coordinates and the area of the face
    """
    
    classifier = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(imgGray, 1.2, 4)
    
    facesAreaList = []
    facesCoordList = []
    
    # Create a list with all the faces detected
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        faceCoord = tools.getBoxCenterCoord2D((x, y, w, h))
        faceArea = w * h
        facesAreaList.append(faceArea)
        facesCoordList.append(faceCoord)
    
    # Keep only the face with the largest area and calculate the z-coordinate 
    if facesCoordList:
        i_max = facesAreaList.index(max(facesAreaList))
        
        mainFaceArea = facesAreaList[i_max]
        mainFaceCoord_xy = facesCoordList[i_max]
        z_coord = math.sqrt(mainFaceArea)
        mainFaceCoord_xyz = mainFaceCoord_xy + [z_coord]
        
        return True, img, mainFaceCoord_xyz
    else:
        return False, img, [0, 0, 0]
    


class PID:
    
    def __init__(self, Kp=0.2, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.last_error = 0
        self.ITerm = 0
        
        self.last_time = time.time()
        
    def run(self, error):
        
        # current_time = time.time()
        # dt = current_time - self.last_time

        PTerm = error
        self.ITerm += error
        # self.ITerm += error*dt

        DTerm = 0.0
        DTerm = error - self.last_error
        # if dt > 0:
        #     DTerm = (error - self.last_error)/dt

        self.last_error = error
        # self.last_time = current_time

        return self.Kp*PTerm + self.Ki*self.ITerm + self.Kd*DTerm
        
    

def trackFace(drone, PID_X, PID_Z, PID_YAW, err, debug = False):
    
    err_X = err[0]
    err_Z = err[1]
    err_YAW = err[2]
    
    # X
    if(err_X != 0):
        com_X = PID_X.run(err_X)
        com_X = int(np.clip(com_X, -100, 100)) 
    else:
        com_X = 0
    
    # Z    
    if(err_Z != 0):
        com_Z = PID_Z.run(err_Z)
        com_Z = int(np.clip(com_Z, -100, 100))     
    else:
        com_Z = 0
       
    # YAW 
    if(err_YAW != 0):
        com_YAW = PID_YAW.run(err_YAW)
        com_YAW = int(np.clip(com_YAW, -100, 100))       
    else:
        com_YAW = 0
        
    if not debug:
        drone.send_rc_control(0, com_X, com_Z, com_YAW)    
    
    return [0, com_X, com_Z, com_YAW]

    
def getError(success, faceCoord, vidCenter, err_X_ref=50, area_mode=True):
    """
        Returns the error on the position of the drone (IN THE DRONE FRAME)
        relatively to the object if it has been detected or not.
        faceCoord is the coordinate of the object IN THE IMAGE FRAME
    
    Args:
        success (_bool_): _description_
        faceCoord (_list_): [x, y, z] corresponding to the position of the object in the image frame
        vidCenter (_list_): [x, y] coordinates of the center of the video in the image frame (constant)
        err_X_ref (float): Reference to compute the x error which will be compared to z coordinate of the face. 
                           Must be the same dimension as faceCoord[2] (e.g. area or distance in meters)
        area_mode (_bool_): true if the reference and faceCoord[2] are areas, false if distances

    Returns:
        _list_: [X, Y, YAW] error in the drone frame
    """
    # print("success : " + str(success))
    # print("faceCoord : " + str(faceCoord))
    
    if success:
        if area_mode:
            err_X = err_X_ref - faceCoord[2]
        else:
            err_X = faceCoord[2] - err_X_ref
            
        err_Z = vidCenter[1] - faceCoord[1]
        err_YAW = faceCoord[0] - vidCenter[0]
        return [err_X, err_Z, err_YAW]
    else:
        return [0, 0, 0]
    
    
def do_trickshot_from_hand_position_id(drone, hand_pose_id):
    
    if hand_pose_id == 0:
        pass
    elif hand_pose_id == 1:
        pass
    elif hand_pose_id == 2:
        drone.flip_forward()
    
    return