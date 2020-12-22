#Usar
#python 01_ObjectDetection_Mask_R-CNN_Inception_v2_COCO_imagem.py
#OpenCV DNN - TensorFlow Mask R-CNN - 0 ~ 90
import numpy as np
import cv2

def prediction(frame, net, CLASSES, COLORS, confidence_lim):
	(frameH, frameW) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	#blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
	numClasses = masks.shape[1]
	numDetections = boxes.shape[2]
	boxesToDraw = []
	for i in range(numDetections):
		box = boxes[0, 0, i]
		mask = masks[i]
		score = box[2]
		if score > 0.4:
			classId = int(box[1])
			left = int(frameW * box[3])
			top = int(frameH * box[4])
			right = int(frameW * box[5])
			bottom = int(frameH * box[6])

			left = max(0, min(left, frameW - 1))
			top = max(0, min(top, frameH - 1))
			right = max(0, min(right, frameW - 1))
			bottom = max(0, min(bottom, frameH - 1))

			boxesToDraw.append([frame, classId, score, left, top, right, bottom])

			classMask = mask[classId]
			classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
			mask = (classMask > 0.4)

			roi = frame[top:bottom+1, left:right+1][mask]
			frame[top:bottom+1, left:right+1][mask] = (0.7 * COLORS[classId] + 0.3 * roi).astype(np.uint8)

			cv2.rectangle(frame, (left, top), (right, bottom), COLORS[classId], 3)
			label = "{}: {:.2f}%".format(CLASSES[classId], score * 100)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[classId], 2)


	return frame


CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush" ]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

use_gpu = 1
if (use_gpu == 1):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
confidence_lim = 0.4

imagem = cv2.imread('bernese-mountain-dog-111878_1280.jpg')
#imagem = cv2.imread('Racionais.jpg')

imagem = prediction(imagem, net, CLASSES, COLORS, confidence_lim)
cv2.imshow("Output", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

