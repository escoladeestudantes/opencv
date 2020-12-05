#Usar
#python 03_object_detection_ssd_mobilenet_video.py
import numpy as np
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

cap = cv2.VideoCapture('Bangkok-30945.mp4')
#cap = cv2.VideoCapture('0')
stop = 0

while (True):
	if(stop == 0):
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
					print("[INFO] {}".format(label))
					cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 4)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
			cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('s'):
		stop = not(stop)
	if key == ord('q'):
		break
cv2.destroyAllWindows()

