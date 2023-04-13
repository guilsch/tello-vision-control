from djitellopy import Tello
import cv2
import time
import numpy as np
import tkinter as tk

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
    
    def updateGains(self, Kp, Ki, Kd):
        if isfloat(Kp):
            self.Kp = float(Kp)
        if isfloat(Ki):
            self.Ki = float(Ki)
        if isfloat(Kd):
            self.Kd = float(Kd)
        
def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False  


def PIDManager(PID, windowName="PID gains"):
    """ Creates a window with PID gains entries. Modifies PID gains when pressing the button.
        You must add window.update() in the loop

    Args:
        PID (PID): PID to be managed
        windowName (str, optional): Name of the window. Default to "PID gains".

    Returns:
        tkinter window
    """
    window = tk.Tk()
    window.title(windowName)

    # Proportional
    label1 = tk.Label(window, text="Proportional:")
    label1.pack(padx=0, pady=0)
    entryP = tk.Entry(window, width=30)
    entryP.pack(padx=10, pady=10)
    entryP.insert(0, PID.Kp)

    # Integral
    label2 = tk.Label(window, text="Integral:")
    label2.pack(padx=0, pady=0)
    entryI = tk.Entry(window, width=30)
    entryI.pack(padx=10, pady=10)
    entryI.insert(0, PID.Ki)

    # Derivative
    label2 = tk.Label(window, text="Derivative:")
    label2.pack(padx=0, pady=0)
    entryD = tk.Entry(window, width=30)
    entryD.pack(padx=10, pady=10)
    entryD.insert(0, PID.Kd)

    def updatePIDGains(P, I, D):
        PID.updateGains(P, I, D)
        
        entryP.delete(0, tk.END)
        entryI.delete(0, tk.END) 
        entryD.delete(0, tk.END) 
        
        entryP.insert(0, PID.Kp)
        entryI.insert(0, PID.Ki)
        entryD.insert(0, PID.Kd)            

    # Button
    button = tk.Button(window, text="Update gains")
    button.pack(padx=10, pady=10)
    button.bind("<Button-1>", lambda event: updatePIDGains(entryP.get(), entryI.get(), entryD.get()))
    
    return window



def controlDrone(drone, err, PID_Z, PID_YAW, PID_X = None, debug = False):
    
    err_X = err[0]
    err_Z = err[1]
    err_YAW = err[2]
        
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
        
    # X
    if PID_X is not None and err_X != 0:
        com_X = PID_X.run(err_X)
        com_X = int(np.clip(com_X, -100, 100)) 
    else:
        com_X = 0    
        
        
    if not debug:
        drone.send_rc_control(0, com_X, com_Z, com_YAW)    
    
    return [0, com_X, com_Z, com_YAW]

    
def getError(success, faceCoord, vidCenter, err_X_ref=50, area_mode=True, mode_3D=True):
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
        mode_3D (_bool_): false if target must be tracked on Z and Yaw only, not X

    Returns:
        _list_: [X, Y, YAW] error in the drone frame
    """
    
    if success:
        if mode_3D:
            if area_mode:
                err_X = err_X_ref - faceCoord[2]
            else:
                err_X = faceCoord[2] - err_X_ref
        else:
            err_X = 0
            
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