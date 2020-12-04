#Usar
#python 04_face_detection_ssd_res10_cff_tf_video_trackbar.py
import numpy as np
import cv2

def nothing(x):
	pass

net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
net2 = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

cv2.namedWindow("Imagem")
cv2.createTrackbar("Confianca", "Imagem", 0, 100, nothing)

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('EduardoMarinho.mp4')

while(True):
    ret, imagem = cap.read()
    (h, w) = imagem.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    net2.setInput(blob)
    detections2 = net2.forward()
    imagem_copia1 = imagem.copy()
    imagem_copia2 = imagem.copy()
    confianca = cv2.getTrackbarPos("Confianca", "Imagem")
    if confianca<1:
        confianca = 1
    confianca = confianca / 100
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confianca:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(imagem_copia1, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(imagem_copia1, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    for i in range(0, detections2.shape[2]):
        confidence2 = detections2[0, 0, i, 2]
        if confidence2 > confianca:
            box = detections2[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence2 * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(imagem_copia2, (startX, startY), (endX, endY),(0, 255, 0), 2)
            cv2.putText(imagem_copia2, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    resultado = np.concatenate((imagem_copia1, imagem_copia2), axis = 1)
    cv2.imshow("Imagem", resultado)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
