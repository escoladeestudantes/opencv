#Usar
#python 01_body_haar_cascade_imagem.py

#Biblioteca
import cv2

fullbody_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')
lowerbody_detector = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
upperbody_detector = cv2.CascadeClassifier('haarcascade_upperbody.xml')

imagem = cv2.imread('pedestrian-zone-347468_640-joergelman-241223.jpg')
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
fullbody = fullbody_detector.detectMultiScale(gray, 1.11, 5)
lowerbody = lowerbody_detector.detectMultiScale(gray, 1.11, 5)
upperbody = upperbody_detector.detectMultiScale(gray, 1.11, 5)
for (x,y,w,h) in fullbody:
	cv2.rectangle(imagem,(x,y),(x+w,y+h),(255,0,0),2)
for (x,y,w,h) in lowerbody:
	cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,255,0),2)
for (x,y,w,h) in upperbody:
	cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('Imagem', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

