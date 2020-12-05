#Usar
#python 01_FaceRecognition_gerar_dataset_viola-jones_video.py

import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#Video
#cap = cv2.VideoCapture('EduardoMarinho01.mp4')
#cap = cv2.VideoCapture('FabioBrazza01.mp4')
#
#cap = cv2.VideoCapture('EduardoMarinho02.mp4')
cap = cv2.VideoCapture('FabioBrazza02.mp4')

#Webcam
#cap = cv2.VideoCapture(0)

face_id = input('\n ID do usuario (valor inteiro):  ')

#25 frames de dois videos totalizando 50 imagens
#contador = 0
contador = 25

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.2, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.imwrite("dataset/User." + str(face_id) + '.' + str(contador) + ".jpg", gray[y:y+h,x:x+w])
		contador += 1
	cv2.imshow('Frame', cv2.resize(frame, (480,500), interpolation = cv2.INTER_AREA))
	k = cv2.waitKey(10) & 0xff 
	if k == 27: # 'ESC' para sair do video
		break
#	elif contador >= 25: # 25 frames do primeiro video
#		break
	elif contador >= 50: # 25 frames do segundo video
		break
cap.release() 
cv2.destroyAllWindows()


