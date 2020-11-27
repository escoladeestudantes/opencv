#Usar
#python carregar_video.py

#Biblioteca
import cv2

#Video
cap = cv2.VideoCapture('../Midias/EduardoMarinho.mp4')
#Webcam
#cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release() 
cv2.destroyAllWindows()

