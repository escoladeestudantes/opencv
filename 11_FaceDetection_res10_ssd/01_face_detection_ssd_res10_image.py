#Usar
#python 01_face_detection_ssd_res10_image.py
import numpy as np
import cv2

net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
#net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

imagem = cv2.imread("Racionais.jpg")
(h, w) = imagem.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
#print(detections)
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    #print(confidence)
    if confidence > 0.18:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #print(box)
        (startX, startY, endX, endY) = box.astype("int")
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(imagem, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(imagem, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.imshow("Output", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

