#Usar
#python 02_ObjectDetection_SSD_MobileNet_COCO_imagens.py
#OpenCV DNN - TensorFlow SSD - 1 ~ 91
import numpy as np
import cv2
import glob
import os

def prediction(imagem, net, CLASSES, COLORS, confidence_lim):
	(h, w) = imagem.shape[:2]
	blob = cv2.dnn.blobFromImage(imagem, size=(300, 300), swapRB=True, crop=False)
	blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 0.007843, (300, 300), 127.5, False)
	#blob = cv2.dnn.blobFromImage(imagem, swapRB=True, crop=False)
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > confidence_lim:
			idx = int(detections[0, 0, i, 1]) - 1 #Porque inicia em 1
			print(idx)
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			print("[INFO] {}".format(label))
			cv2.rectangle(imagem, (startX, startY), (endX, endY),COLORS[idx], 4)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(imagem, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
	print('-------------------')
	return imagem

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush" ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net1 = cv2.dnn.readNetFromTensorflow('frozen_inference_graph_ssd_mb_v1_coco.pb', 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
net1ppn = cv2.dnn.readNetFromTensorflow('frozen_inference_graph_ssd_mb_v1_ppn_coco.pb', 'ssd_mobilenet_v1_ppn_coco.pbtxt')
net2 = cv2.dnn.readNetFromTensorflow('frozen_inference_graph_ssd_mb_v2_coco.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
net3 = cv2.dnn.readNetFromTensorflow('frozen_inference_graph_ssd_mb_v3_coco.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

use_gpu = 1
if (use_gpu == 1):
    net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net1ppn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net1ppn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net3.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net3.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
confidence_lim = 0.4

path = ''
for infile in glob.glob(os.path.join(path, '*.jpg')):
	imagem = cv2.imread(infile)
	imagem_net1 = prediction(imagem.copy(), net1, CLASSES, COLORS, confidence_lim)
	imagem_net1ppn = prediction(imagem.copy(), net1ppn, CLASSES, COLORS, confidence_lim)
	imagem_net2 = prediction(imagem.copy(), net2, CLASSES, COLORS, confidence_lim)
	imagem_net3 = prediction(imagem.copy(), net3, CLASSES, COLORS, confidence_lim)

	cv2.putText(imagem_net1, 'v1', (int(0.90*imagem_net1.shape[1]),int(0.95*imagem_net1.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), cv2.LINE_AA)
	cv2.putText(imagem_net1ppn, 'v1ppn', (int(0.85*imagem_net1ppn.shape[1]),int(0.95*imagem_net1ppn.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), cv2.LINE_AA)
	cv2.putText(imagem_net2, 'v2', (int(0.90*imagem_net2.shape[1]),int(0.95*imagem_net2.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), cv2.LINE_AA)
	cv2.putText(imagem_net3, 'v3', (int(0.90*imagem_net3.shape[1]),int(0.95*imagem_net3.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), cv2.LINE_AA)

	imagem = np.concatenate( ( (np.concatenate((imagem_net1, imagem_net1ppn), axis=1)), (np.concatenate((imagem_net2, imagem_net3), axis=1)) ), axis = 0)

	imagem = cv2.resize(imagem, dsize=(0, 0), fx=0.5, fy=0.5)
	cv2.imshow("Output", imagem)
	cv2.waitKey(0)
cv2.destroyAllWindows()

