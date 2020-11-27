#Usar
#python trackbar_cor_imagem.py

import cv2

def nothing(x):
	pass

imagem = cv2.imread('FabioBrazza.jpg')

cv2.namedWindow("Imagem")

cv2.createTrackbar("Colorida  Cinza", "Imagem", 0, 1, nothing)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

while True:
	s = cv2.getTrackbarPos("Colorida  Cinza", "Imagem")
	if s == 0:
		cv2.imshow("Imagem", imagem)
	else:
		cv2.imshow("Imagem", cinza)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
