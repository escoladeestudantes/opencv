#python 02_FaceRecognition_treinar_classificador_viola-jones_video.py

#Instalar PIL -> pip install pillow

import cv2
import numpy as np
from PIL import Image
import os

# Caminho para a pasta do dataset
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # Converter a imagem para tons de cinza
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Treinando o classificador ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Salvar o modelo .yml
recognizer.write('lbph_classificador/lbph_classificador2.yml')

print("\n [INFO] {0} faces treinadas. Finalizado!".format(len(np.unique(ids))))
