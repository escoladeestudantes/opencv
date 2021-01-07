#Usar
#python OpenCV_Age_and_Gender_Prediction_VGG-16_ETH_Zurich_01_image.py

#https://sefiks.com/2020/09/07/age-and-gender-prediction-with-deep-learning-in-opencv/

import numpy as np
import cv2
import dlib

#model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
#pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")
#age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_imdb_wiki.caffemodel")

#model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
#pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
gender_model = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")

use_gpu = 1
if (use_gpu == 1):
    age_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    age_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    gender_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    gender_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

detector = dlib.get_frontal_face_detector() #HOG
#detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat") #MMOD

output_indexes = np.array([i for i in range(0, 101)])

image = cv2.imread('CesarMc.jpg')
faces = detector(image, 0)
for face in faces:
    x1,x2,y1,y2 = face.left(), face.right(), face.top(),face.bottom()
    #x1,x2,y1,y2 = face.rect.left(), face.rect.right(), face.rect.top(),face.rect.bottom() #MMOD
    detected_face = image[y1:y2, x1:x2]
    detected_face = cv2.resize(detected_face, (224, 224))
    img_blob = cv2.dnn.blobFromImage(detected_face) #caffe model expects (1, 3, 224, 224) shape input
    #---------------------------
    age_model.setInput(img_blob)
    age_dist = age_model.forward()[0]
    apparent_predictions = round(np.sum(age_dist * output_indexes), 2)
    print("Apparent age: ",apparent_predictions)

    gender_model.setInput(img_blob)
    gender_class = gender_model.forward()[0]
    gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
    print("Gender: ", gender)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image, str(apparent_predictions), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 6)
    cv2.putText(image, str(apparent_predictions), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
    cv2.putText(image, str(gender), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 6)
    cv2.putText(image, str(gender), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
    cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
