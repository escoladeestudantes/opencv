#Usar
#python OpenCV_IP_Webcam_01_VideoCapture_frames.py

import cv2
import numpy as np

url='http://192.168.0.144:8080/video'

def receber_imagem(cap):
    c = 0
    while True:
        ret, img = cap.read()
        if (ret==True):
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
        video = cv2.VideoCapture(url)
        status = receber_imagem(video)
        if status == 0:
            print('Webcam offline.')
    if (tentativa == '2'):
        break
cv2.destroyAllWindows()
