#Usar
#python 05_ObjectDetection_YOLO_v4_videos.py
import numpy as np
import cv2

yolov4_320 = cv2.VideoCapture('Bangkok-30945-yolov4-320.avi')
yolov4_416 = cv2.VideoCapture('Bangkok-30945-yolov4-416.avi')
yolov4_512 = cv2.VideoCapture('Bangkok-30945-yolov4-512.avi')
yolov4_608 = cv2.VideoCapture('Bangkok-30945-yolov4-608.avi')

stop = 0

while (True):
	if(stop == 0):
		ret320, frame320 = yolov4_320.read()
		ret416, frame416 = yolov4_416.read()
		ret512, frame512 = yolov4_512.read()
		ret608, frame608 = yolov4_608.read()
		if (ret320 and ret416 and ret512 and ret608):
			frame = np.vstack( ( np.hstack((frame320,frame416)),  np.hstack((frame512,frame608)) ) )
			frame = cv2.resize(frame, (int(0.85 * frame.shape[1]), int(0.85 * frame.shape[0])))
			cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('s'):
		stop = not(stop)
	if key == ord('q'):
		break
cv2.destroyAllWindows()

