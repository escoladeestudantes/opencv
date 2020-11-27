#Usar
#python trackbar_binarizacao_imagem.py

import cv2
import numpy as np

def nothing(x):
	pass

imagem = cv2.imread('FabioBrazza.jpg')

cv2.namedWindow("Imagem")
cv2.createTrackbar("Lim_inf", "Imagem", 0, 254, nothing)
cv2.createTrackbar("Lim_sup", "Imagem", 1, 255, nothing)
cv2.createTrackbar("Colorida Cinza", "Imagem", 0, 1, nothing)
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

while True:
	lim_inf = cv2.getTrackbarPos("Lim_inf", "Imagem")
	lim_sup = cv2.getTrackbarPos("Lim_sup", "Imagem")
	s = cv2.getTrackbarPos("Colorida Cinza", "Imagem")
	edges = cv2.Canny(cinza,lim_inf,lim_sup)
	if s == 0:
		#edges passa a ter a mesma estrutura (RGB) de imagem
		edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
		resultado = np.concatenate((imagem, edges), axis=1)
	else:
		resultado = cv2.hconcat([cinza, edges])
	cv2.imshow("Imagem", resultado)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
