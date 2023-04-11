import cv2

while True:
  
    print('rien')
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        print('arrêt')
        break 