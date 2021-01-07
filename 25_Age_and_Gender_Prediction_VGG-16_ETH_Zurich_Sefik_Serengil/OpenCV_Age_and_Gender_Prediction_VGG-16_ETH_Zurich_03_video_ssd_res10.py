#Usar
#python OpenCV_Age_and_Gender_Prediction_VGG-16_ETH_Zurich_03_video_ssd_res10.py
import numpy as np
import cv2

net_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#net_model = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')

#model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
#pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")
#age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_imdb_wiki.caffemodel")

#model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
#pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
gender_model = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")

use_gpu = 1
if (use_gpu == 1):
    net_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    age_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    age_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    gender_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    gender_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

output_indexes = np.array([i for i in range(0, 101)])

cap = cv2.VideoCapture('Gigante_no_Mic_Apendice.mp4')
#cap = cv2.VideoCapture(0)
stop = 0
ret=True
confianca = 0.7

while (ret==True):
    if(stop == 0):
        ret, frame = cap.read()
        if (ret==True):
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
            net_model.setInput(blob)
            detections = net_model.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confianca:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    x1,x2,y1,y2 = startX, endX, startY, endY 
                    detected_face = frame[y1:y2, x1:x2]
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
    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(frame, str(apparent_predictions), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 6)
                    cv2.putText(frame, str(apparent_predictions), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)
                    cv2.putText(frame, str(gender), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 6)
                    cv2.putText(frame, str(gender), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3)

                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)                    
                    #text = "{:.2f}%".format(confidence * 100)
                    #y = startY - 10 if startY - 10 > 10 else startY + 10
                    #cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow("Imagem", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        stop = not(stop)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
