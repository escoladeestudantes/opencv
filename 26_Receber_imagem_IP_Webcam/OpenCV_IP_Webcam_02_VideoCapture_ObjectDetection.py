#Usar
#python OpenCV_IP_Webcam_02_VideoCapture_ObjectDetection.py
#OpenCV DNN - TensorFlow Faster R-CNN - 1 ~ 91
import numpy as np
import cv2

def prediction(imagem, net, CLASSES, COLORS, confidence_lim):
	(h, w) = imagem.shape[:2]
	blob = cv2.dnn.blobFromImage(imagem, swapRB=True, crop=False)
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > confidence_lim:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			#print("[INFO] {}".format(label))
			cv2.rectangle(imagem, (startX, startY), (endX, endY),COLORS[idx], 4)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(imagem, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
	return imagem

def receber_imagem(cap):
    while True:
        ret, img = cap.read()
        if (ret==True):
            img = prediction(img, net, CLASSES, COLORS, confidence_lim)
            cv2.imshow('Imagem',img)
        else:
            return 0   
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            return 1

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush" ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'faster_rcnn_resnet50_coco_2018_01_28.pbtxt')

use_gpu = 1
if (use_gpu == 1):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
confidence_lim = 0.6

url='http://192.168.0.144:8080/video'
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
