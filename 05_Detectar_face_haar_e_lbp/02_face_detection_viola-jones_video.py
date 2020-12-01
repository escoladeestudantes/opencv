#Usar
#python 02_face_detection_viola-jones_video.py

#Biblioteca
import cv2

#Modelo de detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Video
cap = cv2.VideoCapture('EduardoMarinho.mp4')
#Webcam
#cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.2, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release() 
cv2.destroyAllWindows()

