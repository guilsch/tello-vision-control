import colorsys
import cv2
import numpy as np
import copy
import itertools
import csv


def trackerCreate(tracker_type):
    
    print(cv2.__version__)
    
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
            tracker = cv2.TrackerMOSSE_create()
            print("Tracker type must be one of the following : 'BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE', set to MOSSE tracker")

    return tracker

def getVideoCenterCoord(video):
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center = (int(width / 2), int(height / 2))
    
    return center

def getBoxCenterCoord3D(box, initWidth):
    
    x = int(box[0] + box[2]/2)
    y = int(box[1] + box[3]/2)
    rel_z = round(box[2] / initWidth - 1, 3)
    
    return (x, y, rel_z)

def getBoxCenterCoord2D(box):
    
    x = int(box[0] + box[2]/2)
    y = int(box[1] + box[3]/2)
    
    return (x, y)


def createColorGrading(param):
    
    param = (param + 1) / 2
    
    start_color = (255, 0, 0)
    end_color = (0, 0, 255)

    color = np.array(start_color) * (1 - param) + np.array(end_color) * param
    
    return color


def binFilter(img, thresh):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binImg = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    return binImg


def getShapeCenter(binImg):
    # Calculer les moments de l'objet
    M = cv2.moments(binImg)

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    return (x, y)

def blurBG(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    frame = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)
    
    return frame


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


##### Keypoints #####
#####################

def getCentroidFromKeypoints(keypoints):
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
    
    # find the keypoints and descriptors
    kp_ref, des_ref = descriptor.detectAndCompute(img_ref, None)
    kp_test, des_test = descriptor.detectAndCompute(img_test, None)

    print(f"kp_test len : {str(len(kp_test))}")
    print(f"kp_ref len : {str(len(kp_ref))}")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_ref, des_test, k=2)
    
    print(f"matches len : {str(len(matches))}")

    # Keep only good matches
    good = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)

    print(f"good matches len : {str(len(good))}")

    good_kp_ref = [kp_ref[m.queryIdx] for m in good]
    good_kp_test = [kp_test[m.queryIdx] for m in good]
    
    return good_kp_test, good_kp_ref, good


##### Mobilenet #####
#####################

def getClassNames(fileAdress):
    """ Return list with class names from the file specified by fileAdress
    
    """
    classNames= []
    with open(fileAdress, "r") as f:
        classNames = [ligne.strip() for ligne in f]
        
    return classNames

def getClassNameFromId(classNames, id):
    return classNames[int(id)-1]

def getDetectionsList(classIds, confs, bbox, classNames, filteredClass = None):
    """Provide lists of (box, conf, id). filteredList is for object class belonging in filteredClass, 
        othersList is for all the other objects. If filteredClass is not given, it returns both lists with all objects inside
    """
    filteredList = []
    othersList = []
    
    for id, conf, box in zip(classIds, confs, bbox):
        if filteredClass == None or getClassNameFromId(classNames, id) in filteredClass:
            filteredList.append((box, conf[0], id[0]))
        
        if filteredClass == None or not getClassNameFromId(classNames, id) in filteredClass:
            othersList.append((box, conf[0], id[0]))
            
    return filteredList, othersList


##### MEDIAPIPE #####
#####################
def getPoseCoord(results, num_pose, w_image, h_image):
    """Returns a list containing the position of the pose in the image.

    Args:
        results (_type_): the object given by the pose.process() method, the result of the pose detection
        num_pose (_type_): the number 0-32 corresponding to a particular part of the pose landmark
        w_image (_type_): width of the image reference for the coordinates
        h_image (_type_): height of the image reference for the coordinates

    Returns:
        _type_: list [x, y, z, visibility] with x, y the coordinate of the particular pose in the image coordinates
    """
    num_pose_count = 0
    coord = [None, None, None, None]
    landmarks = results.pose_landmarks.landmark
    
    if landmarks:
        for data_point in landmarks:
            
            if num_pose_count == num_pose:
                x, y = convertNormCoordToImageCoord(data_point.x, data_point.y, w_image, h_image)
                coord = [x, y, data_point.z, data_point.visibility]
                return coord
            
            num_pose_count += 1
        
    return coord


def convertNormCoordToImageCoord(x, y, w_image, h_image):
    """Convert coordinates x and y from 0.0-1.0 reference to image-size reference (e.g. 0-720)

    Args:
        x (_type_): 0.0 to 1.0 x coordinate
        y (_type_): 0.0 to 1.0 y coordinate
        w_image (_type_): image size reference
        h_image (_type_): image size reference

    Returns:
        _type_: x, y in the image size reference
    """
    return int(x*w_image), int(y*h_image)


def get_labels_list(adress):
    """Load labels for hand gesture classification in a list from a csv file

    Args:
        adress (String): adress of the csv file containing labels

    Returns:
        (list) : list with all labels
    """
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