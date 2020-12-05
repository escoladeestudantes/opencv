#Usar
#python 02_object_detection_ssd_mobilenet_imagens.py
import numpy as np
import cv2
import glob
import os

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

path = ''
for infile in glob.glob(os.path.join(path, '*.jpg')):
	imagem = cv2.imread(infile)
	(h, w) = imagem.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
	    confidence = detections[0, 0, i, 2]
	    if confidence > 0.7:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(imagem, (startX, startY), (endX, endY),COLORS[idx], 4)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(imagem, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
	cv2.imshow("Output", imagem)
	cv2.waitKey(0)
cv2.destroyAllWindows()

