#Usar
#python OpenCV_IP_Webcam_01_urllib_frames.py

import urllib.request
import cv2
import numpy as np

# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://192.168.0.144:8080/shot.jpg'

def receber_imagem():
    c = 0
    while True:
        # Use urllib to get the image from the IP camera
        imgResp = urllib.request.urlopen(url)
        # Numpy to convert into a array
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        #print(imgNp)
        if (imgNp != []):
        # Finally decode the array to OpenCV usable format ;) 
            img = cv2.imdecode(imgNp,-1)
            #img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)    
            # put the image on screen
            cv2.imshow('Imagem',img)
            c+=1
            print(c)
        else:
            return 0
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            return 1

tentativa = 0
while(True):
    tentativa = input('1 para iniciar | 2 para finalizar: ')
    if (tentativa == '1'):
        status = receber_imagem()
        if status == 0:
            print('Webcam offline.')
    if (tentativa == '2'):
        break
cv2.destroyAllWindows()
