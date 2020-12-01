#Usar
#python 03_face_detection_viola-jones_imagem_trackbar.py

#Biblioteca
import cv2

def nothing(x):
	pass

#Carregar detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Carregar imagem
imagem = cv2.imread('Racionais.jpg')
#Imagem RGB para tons de cinza
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("frame")

cv2.createTrackbar("Diminuicao", "frame", 1100, 1800, nothing)
cv2.createTrackbar("MinVizinhos", "frame", 1, 10, nothing)

while True:
	imagem_copia = imagem.copy()
	diminuicao = cv2.getTrackbarPos("Diminuicao", "frame")
	if diminuicao<1050:
		diminuicao = 1050
	diminuicao = diminuicao / 1000
	vizinhos = cv2.getTrackbarPos("MinVizinhos", "frame")
	faces = face_detector.detectMultiScale(gray, diminuicao, vizinhos)
	#Para cada face encontrada
	for (x,y,w,h) in faces:
		#Desenhar um retangulo
		cv2.rectangle(imagem_copia,(x,y),(x+w,y+h),(255,0,0),2)
	#Mostrar o resultado final
	cv2.imshow('frame',imagem_copia)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()


