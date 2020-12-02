#Usar
#python 02_body_hog_imagem.py

#Biblioteca
import cv2

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imagem = cv2.imread('pedestrian-zone-347468_640-joergelman-241223.jpg')
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
(regions, _) = hog.detectMultiScale(imagem, winStride=(4, 4), padding=(4, 4), scale=1.15)
for (x,y,w,h) in regions:
	cv2.rectangle(imagem,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow('Imagem', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

