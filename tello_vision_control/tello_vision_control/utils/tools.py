""" This file contains all methods used to find objects in a video. They can all be used without drone.

"""

import cv2
import numpy as np
import copy
import itertools
import csv
import os
import tello_vision_control
import math


######## PARAMETERS #######
###########################

package_dir = os.path.dirname(tello_vision_control.__file__)


########## TESTS ##########
###########################

def test():
    print("test ok")
    return


########## UTILS ##########
###########################

def getVideoCenterCoord(video):
    """ Returns the center of the video in pixel units: (x, y)
    """
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center = (int(width / 2), int(height / 2))
    
    return center

def getBoxCenterCoord3D(box, initWidth):
    """ Returns the coordinate of the center of the box in 3-dimensions. 
    The z-dimension is computed by comparing initial width with the width of the box. It is a positive
    value if the box is closer than its original position and negative if it is further. 

    Args:
        box : Bounding box returned by tracker.update() method of opencv
        initWidth (int): Width reference for the object

    Returns:
        tuple: (x, y, z) integer coordinates in pixel units
    """
    x = int(box[0] + box[2]/2)
    y = int(box[1] + box[3]/2)
    rel_z = round(box[2] / initWidth - 1, 3)
    
    return (x, y, rel_z)

def getBoxCenterCoord2D(box):
    """ Returns the coordinate of the center of the box in 2-dimensions.

    Args:
        box : Bounding box returned by tracker.update() method of opencv

    Returns:
        tuple: (x, y) integer coordinates in pixel units
    """
    x = int(box[0] + box[2]/2)
    y = int(box[1] + box[3]/2)
    
    return (x, y)


def createColorGrading(param):
    """ Returns a color generated with one parameter. Used to compute a color from the z coordinate of an 
    object to show the depth of the object.

    Args:
        param (float): z-coordinate of an object

    Returns:
        color
    """
    param = (param + 1) / 2
    
    start_color = (255, 0, 0)
    end_color = (0, 0, 255)

    color = np.array(start_color) * (1 - param) + np.array(end_color) * param
    
    return color


def binFilter(img, thresh):
    """ Returns a binary image computed with a threshold

    Args:
        img (image): originale image
        thresh : threshold between 0 to 255

    Returns:
        image: binary image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binImg = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    return binImg


def getShapeCenter(binImg):
    """ Compute center of a shape of an object from binary image.
    """
    M = cv2.moments(binImg)

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    return (x, y)

def blurBG(frame):
    """ Returns image with blurred background
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    frame = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)
    
    return frame

###### OPENCV TRACKER #####
###########################

def trackerCreate(tracker_type):
    """ Initialize a tracker from the OpenCV library. Tracker can be one of the following :
        'BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE'.
        Check https://docs.opencv.org/3.4/d9/df8/group__tracking.html for more information.

    Args:
        tracker_type (String): tracker name 

    Returns:
        opencv tracker: tracker
    """
    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    if int(major_ver) < 4 and int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        else:
            tracker = cv2.TrackerMedianFlow_create()
            print("Tracker type must be one of the following : 'BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE', set to MEDIANFLOW tracker")

    return tracker


########## YOLO ###########
###########################

def detectWithYOLO(net, image, confThresh):
    
    (H, W) = image.shape[:2]
    
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layerNames)
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confThresh:
                # Scale the bounding box coordinates relatively to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # # Find indexes corresponding to non-filtered boxes            
    # filteredIndex = cv2.dnn.NMSBoxes(boxes, confidences, confThresh, 0.3).flatten()
    # print(filteredIndex)
    
    # # Filter detections
    # boxes = [boxes[i] for i in filteredIndex]
    # confidences = [confidences[i] for i in filteredIndex]
    # classIDs = [classIDs[i] for i in filteredIndex]
    
    return classIDs, confidences, boxes


######## KEYPOINTS ########
###########################

def getCentroidFromKeypoints(keypoints):
    """ Returns mass center of keypoints cloud
    """
    num_keypoints = len(keypoints)
    
    if num_keypoints != 0:
        x_sum = 0
        y_sum = 0
        for kp in keypoints:
            x_sum += kp.pt[0]
            y_sum += kp.pt[1]

        x_mean = int(x_sum / num_keypoints)
        y_mean = int(y_sum / num_keypoints)

        return (x_mean, y_mean)
    
    return (0, 0)

def getKeypoints(img_ref, img_test, descriptor, thresh):
    """ Returns keypoints that match with the keypoints of the reference image 
    """
    # find the keypoints and descriptors
    kp_ref, des_ref = descriptor.detectAndCompute(img_ref, None)
    kp_test, des_test = descriptor.detectAndCompute(img_test, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_ref, des_test, k=2)

    # Keep only good matches
    good = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)

    print(f"Good matches len : {str(len(good))}")

    good_kp_ref = [kp_ref[m.queryIdx] for m in good]
    good_kp_test = [kp_test[m.queryIdx] for m in good]
    
    return good_kp_test, good_kp_ref, good


##### Mobilenet #####
#####################

def initDetectionModel(weightsPath=None, configPath=None):
    """ Initialize a detection model used to detect object. If no weightsPath or configPath is given,
    default path will be used.
    
    """
    if weightsPath is None:
        weightsPath =  os.path.join(package_dir, 'data/ssd_mobilenet_v3_files', 
                                    'frozen_inference_graph.pb')
    
    if configPath is None:
        configPath =  os.path.join(package_dir, 'data/ssd_mobilenet_v3_files', 
                                   'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    
    return net

def getClassNames(fileAdress=None):
    """ Return list with class names from the file specified by fileAdress.
    Default names files will be used if none is given.
    
    """
    if fileAdress is None:
        fileAdress =  os.path.join(package_dir, 'data/ssd_mobilenet_v3_files', 
                                   'coco.names')
    
    classNames= []
    with open(fileAdress, "r") as f:
        classNames = [ligne.strip() for ligne in f]
        
    return classNames

def getClassNameFromId(classNames, id):
    return classNames[int(id)-1]

def getDetectionsList(classIds, confs, bbox, classNames, filteredClass = None):
    """ Provide lists of (box, conf, id). filteredList is a list with objects belonging in filteredClass, 
        othersList is a list with all the other objects. If filteredClass is not given, it returns both lists 
        with all objects inside.
        This method gets objects that will be processed by a tracker from objects that has been detected by a 
        neural network trained for detection.
    """
    filteredList = []
    othersList = []
    
    for id, conf, box in zip(classIds, confs, bbox):
        if filteredClass == None or getClassNameFromId(classNames, id) in filteredClass:
            filteredList.append((box, conf[0], id[0]))
        
        if filteredClass == None or not getClassNameFromId(classNames, id) in filteredClass:
            othersList.append((box, conf[0], id[0]))
            
    return filteredList, othersList


######## MEDIAPIPE ########
###########################

def getLandmarkCoord(pose, num_landmark, w_image, h_image, head_size_ref = 20, hor_cam_ang = 82.6):
    """Returns a list containing the position of the pose in the image.

    Args:
        pose (_type_): the object given by the pose.process() method, the result of the pose detection
        num_pose (_type_): the number 0-32 corresponding to a particular part of the pose landmark
        w_image (_type_): width of the image reference for the coordinates
        h_image (_type_): height of the image reference for the coordinates
        head_size_ref (_type): Head size reference between the ears in centimeters

    Returns:
        _type_: list [x, y, z, visibility] with x, y the coordinate of the particular landmark in the image coordinates
    """
    coord = [None, None, None]
    success = False
    
    # horiz_cam_angle = abs(2 * math.atan(math.tan(diag_cam_ang / 2) * (w_image / h_image)))
    
    if pose.pose_landmarks:
        landmarks = pose.pose_landmarks.landmark
        
        for (num_landmark_count, data_point) in enumerate(landmarks):
            
            if num_landmark_count == 7:
                left_ear_coord = [data_point.x*w_image, data_point.y*h_image]  
            elif num_landmark_count == 8:
                right_ear_coord = [data_point.x*w_image, data_point.y*h_image]
            
            if num_landmark_count == num_landmark:
                coord[0] = int(data_point.x*w_image)
                coord[1] = int(data_point.y*h_image)
    
        if left_ear_coord[0] is not None and right_ear_coord[0] is not None:
            head_size_pixel = math.hypot(left_ear_coord[0] - right_ear_coord[0], left_ear_coord[1] - right_ear_coord[1])
            depth = round((head_size_ref * w_image / 2) / (head_size_pixel * math.tan(hor_cam_ang)), 3)
            coord[2] = min(depth, 300)
            
        success = True
      
    return success, coord


# def convertNormCoordToImageCoord(x, y, w_image, h_image):
#     """Convert coordinates x and y from 0.0-1.0 reference to image-size reference (e.g. 0-720)

#     Args:
#         x (_type_): 0.0 to 1.0 x coordinate
#         y (_type_): 0.0 to 1.0 y coordinate
#         w_image (_type_): image size reference
#         h_image (_type_): image size reference

#     Returns:
#         _type_: x, y in the image size reference
#     """
#     return int(x*w_image), int(y*h_image)


def get_labels_list(adress=None):
    """Load labels for hand gesture classification in a list from a csv file

    Args:
        adress (String): adress of the csv file containing labels

    Returns:
        (list) : list with all labels
    """
    if adress is None:
        adress = os.path.join(package_dir, 'data', 'keypoint_classifier_label.csv')
    
    with open(adress,
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
        
    return keypoint_classifier_labels


def pre_process_landmark(landmark_list):
    """ Transforms landmark list [[x, y], ...] to a format fitting the classification model
        This method is an adaptation of the pre_process_landmark() method in the app.py 
        file of the following repo :
        https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe

    Args:
        landmarks (landmarks list): list containing a landmarks coordinates list : [[x, y], [x, y], ...] 
        returned by get_landmark_coord_list().

    Returns:
        list: List containing relative coordinates normalized and flatten 
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Get coordinates relatively to the 0 hand landmark (base of the hand palm)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Flatten the coordinates list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalize the values
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def get_landmarks_box(landmarks, image_width, image_height):
    """ Calculates bounding box coordinates around a landmarks group (hand)
        This method is an adaptation of the calc_bounding_rect() method in the app.py 
        file of the following repo :
        https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe

    Args:
        landmarks (landmarks): landmarks corresponding to one hand
        image_width (int): width of the image
        image_height (int): height of the image

    Returns:
        list: bounding box
    """

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def get_landmark_coord_list(landmarks, image_width, image_height):
    """ Calculates coordinates in the image of every landmark of one landmarks group (hand)
        This method is an adaptation of the calc_landmark_list() method in the app.py 
        file of the following repo :
        https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe

    Args:
        landmarks (landmarks): landmarks corresponding to one hand
        image_width (int): width of the image
        image_height (int): height of the image

    Returns:
        list: landmark coordinates list
    """

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point