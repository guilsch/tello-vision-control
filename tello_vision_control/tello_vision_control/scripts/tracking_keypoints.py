import cv2
from tello_vision_control.utils import tools

##### Init video
# video = cv2.VideoCapture("videos/xy_slow.mp4")
# # video = cv2.VideoCapture(0)

# vidCenter = tools.getVideoCenterCoord(video)

##### Create descriptor
descriptor = cv2.ORB_create()
thresh = 0.85

##### Define reference object
# img_ref = cv2.imread("images/face_ref.jpg")
# _, frame1 = video.read()
# r = cv2.selectROI("select the area", frame1)
# img_ref = frame1[int(r[1]):int(r[1]+r[3]), 
#                       int(r[0]):int(r[0]+r[2])]

img_ref = cv2.imread("images/apple_ref.png")
img_test = cv2.imread("images/apple.png")

# Get matched keypoints
kp_test, kp_ref, matches = tools.getKeypoints(img_ref, img_test, descriptor, thresh)

# Calculate centroid
centroid = tools.getCentroidFromKeypoints(kp_test)

# Draw matches
img_matches = cv2.drawMatches(img_ref,kp_ref,img_test,kp_test, matches,None)
# img_match = cv2.drawMatchesKnn(img_ref, kp_ref, img_test, kp_test, good, None)

# Draw      
img_ref_kp = cv2.drawKeypoints(img_test, kp_test, None)
img_ref_kp = cv2.circle(img_ref_kp, centroid, 5, (0, 0, 255), -1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow("image", img_matches)
cv2.waitKey(0)

# ##### Real-time tracking
# while True:

#     success, frame = video.read()
#     if not success:
#         break

#     # Get matched keypoints
#     matched_kp = tools.getKeypoints(img_ref, frame, descriptor, thresh)

#     # Calculate centroid
#     centroid = tools.getCentroidFromKeypoints(matched_kp)
    
#     # Draw      
#     img_ref_kp = cv2.drawKeypoints(frame, matched_kp, None)
#     img_ref_kp = cv2.circle(img_ref_kp, centroid, 5, (0, 0, 255), -1)
    
#     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#     cv2.imshow("image", img_ref_kp)
    
#     # Exit if ESC pressed
#     k = cv2.waitKey(1) & 0xff
#     if k == 27 : break
    
# video.release()