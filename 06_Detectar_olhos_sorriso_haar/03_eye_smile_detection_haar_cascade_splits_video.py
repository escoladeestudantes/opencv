#Usar
#python 03_eye_smile_detection_haar_cascade_splits_video.py

#Biblioteca
import cv2

#Modelo de detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
lefteye_detector = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
righteye_detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
#Video
cap = cv2.VideoCapture('EduardoMarinho.mp4')
#Webcam
#cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.18, 5)
	#Em cada face encontrada
	for (x,y,w,h) in faces:
		#Seleciona-se a face - regiao de interesse (ROI)
		faceROI = gray[y:y+h,x:x+w]
		#Apenas na ROI aplica-se o detector de olhos e sorriso
		lefteye = lefteye_detector.detectMultiScale(faceROI, 1.2, 5)
		righteye = righteye_detector.detectMultiScale(faceROI, 1.2, 5)
		smile = smile_detector.detectMultiScale(faceROI, 1.8, 5)
		for (x2,y2,w2,h2) in lefteye:
			eye_center = (x + x2 + w2//2, y + y2 + h2//2)
			radius = int(round((w2 + h2)*0.25))
			cv2.circle(frame, eye_center, radius, (0, 255, 0 ), 4)
		for (x3,y3,w3,h3) in righteye:
			eye_center = (x + x3 + w3//2, y + y3 + h3//2)
			radius = int(round((w3 + h3)*0.25))
			cv2.circle(frame, eye_center, radius, (0,255,255), 4)
		for (x4,y4,w4,h4) in smile:
			cv2.rectangle(frame,(x+x4,y+y4),(x+x4+w4,y+y4+h4),(0,0,255),3)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release() 
cv2.destroyAllWindows()

