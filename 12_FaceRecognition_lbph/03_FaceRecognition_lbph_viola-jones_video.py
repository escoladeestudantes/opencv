#Usar
#python 03_FaceRecognition_lbph_viola-jones_video.py

import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read('lbph_classificador/lbph_classificador1.yml')
recognizer.read('lbph_classificador/lbph_classificador2.yml')
cascadePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

id = 0

# Adicionar os nomes conforme a ID definida no dataset 
names = ['Nenhum', 'EduardoMarinho', 'FabioBrazza'] 

#Videos que geraram o dataset
#cam = cv2.VideoCapture('FabioBrazza01.mp4')
#cam = cv2.VideoCapture('EduardoMarinho01.mp4')
#
#cam = cv2.VideoCapture('FabioBrazza02.mp4')
#cam = cv2.VideoCapture('EduardoMarinho02.mp4')
#Videos diferentes
cam = cv2.VideoCapture('EduardoMarinho03.mp4')
#cam = cv2.VideoCapture('FabioBrazza03.mp4')

while True:
    ret, frame =cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( gray, scaleFactor = 1.2, minNeighbors = 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Confidence == 0 significa total certeza
        if (confidence < 100):
            if (confidence < 65): #Maior do que 35%
            	id = names[id]
            	confidence = "  {0}%".format(round(100 - confidence))
            else:
            	id = "Desconhecido"
            	confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(frame, str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)  
    cv2.imshow('Imagem',cv2.resize(frame, (650,780), interpolation = cv2.INTER_AREA)) 
    k = cv2.waitKey(1) & 0xff 
    if k == 27: #'ESC' para sair
        break
cam.release()
cv2.destroyAllWindows()
