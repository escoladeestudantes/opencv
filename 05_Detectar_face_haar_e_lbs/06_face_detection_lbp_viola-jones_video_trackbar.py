#Usar
#python 06_face_detection_lbp_viola-jones_video_trackbar.py

#Biblioteca
import cv2

def nothing(x):
	pass

#Carregar detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_detector2 = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
#Video
cap = cv2.VideoCapture('EduardoMarinho.mp4')
#Webcam
#cap = cv2.VideoCapture(0)

cv2.namedWindow("frame")

cv2.createTrackbar("Diminuicao", "frame", 1100, 1800, nothing)
cv2.createTrackbar("MinVizinhos", "frame", 1, 10, nothing)

while True:
	ret, imagem = cap.read()
	#Imagem RGB para tons de cinza
	gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	diminuicao = cv2.getTrackbarPos("Diminuicao", "frame")
	if diminuicao<1050:
		diminuicao = 1050
	diminuicao = diminuicao / 1000
	vizinhos = cv2.getTrackbarPos("MinVizinhos", "frame")
	faces = face_detector.detectMultiScale(gray, diminuicao, vizinhos)
	#Para cada face encontrada
	for (x,y,w,h) in faces:
		#Desenhar um retangulo
		cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,0,255),2)
	#Mostrar o resultado final
	faces2 = face_detector2.detectMultiScale(gray, diminuicao, vizinhos)
	#Para cada face encontrada
	for (x,y,w,h) in faces2:
		#Desenhar um retangulo
		cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,255,0),2)
	#Mostrar o resultado final
	cv2.imshow('frame',imagem)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
