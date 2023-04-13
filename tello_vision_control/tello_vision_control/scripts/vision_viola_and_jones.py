from tello_vision_control import vision_tools
import cv2

########## SETUP ##########
###########################

#### Cascade Classifier
classifier = vision_tools.createHaarCascadeClassifier()

##### Init video
video_adress = None
if video_adress is not None:
    # Video read from adress provided
    video = cv2.VideoCapture(video_adress)
else:
    # Video read from camera of the device
    video = cv2.VideoCapture(0)

w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
vidCenter = (int(w/2), int(h/2))


########## LOOP ###########
###########################
run = True
while run:
    
    # Exit if ENTER pressed
    k = cv2.waitKey(1) & 0xff
    if k == 13 :
        run = False
    
    success, frame = video.read()
    if not success:
        break

    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(imgGray, 1.2, 4)
    
    success, frame, faceCoord = vision_tools.showAndFilterFaces(frame, faces)
    
    cv2.imshow("Tello camera", frame)
    cv2.waitKey(1)