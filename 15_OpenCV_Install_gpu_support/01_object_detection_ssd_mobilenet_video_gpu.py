#Usar
#python 01_object_detection_ssd_mobilenet_video_gpu.py
import numpy as np
import cv2
from imutils.video import FPS

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

#use_gpu = 0
use_gpu = 1
if use_gpu == 1:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture('Bangkok-30945.mp4')
#cap = cv2.VideoCapture('0')

fps = FPS().start()
ret = True
while (ret == True):
	ret, frame = cap.read()
	if (ret == True):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.7:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				#print("[INFO] {}".format(label))
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 4)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
		cv2.imshow("Frame", frame)
		fps.update()
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.waitKey(0)
cv2.destroyAllWindows()

