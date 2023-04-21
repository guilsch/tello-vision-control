import cv2
from tello_vision_control.utils import vision_tools

##### Init video
video = cv2.VideoCapture(0)
vidCenter = vision_tools.getVideoCenterCoord(video)

##### Init reference image
im_ref_address = "images/apple_ref.png"
img_ref = cv2.imread(im_ref_address)

##### Create descriptor
descriptor = cv2.ORB_create()
thresh = 0.85


##### Real-time tracking
while True:

    success, frame = video.read()
    if not success:
        break

    # Get matched keypoints
    matched_kp = vision_tools.getKeypoints(img_ref, frame, descriptor, thresh)

    # Calculate centroid
    centroid = vision_tools.getCentroidFromKeypoints(matched_kp)
    
    # Draw      
    img_ref_kp = cv2.drawKeypoints(frame, matched_kp[0], None)
    img_ref_kp = cv2.circle(img_ref_kp, centroid, 5, (0, 0, 255), -1)
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", img_ref_kp)
    
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
    
video.release()