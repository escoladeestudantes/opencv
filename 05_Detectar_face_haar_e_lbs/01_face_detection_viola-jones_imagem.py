#Usar
#python 01_face_detection_viola-jones_imagem.py

#Biblioteca
import cv2

#Carregar detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Carregar imagem
imagem = cv2.imread('Racionais.jpg')
#Imagem RGB para tons de cinza
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#Parametros obrigatorios
#Imagem cinza porque o classificador foi treinado assim (pode ser colorida)
#Reducao da imagem na busca de faces. Maior = mais rapido
#Quantidade de vizinhos para cada candidato de face
faces = face_detector.detectMultiScale(gray, 1.2, 5)
#Para cada face encontrada
for (x,y,w,h) in faces:
    #Desenhar um retangulo
    print(x,y,w,h)
    cv2.rectangle(imagem,(x,y),(x+w,y+h),(255,0,0),2)
#Mostrar o resultado final
cv2.imshow('Imagem',imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Referencia
#https://docs.opencv.org/4.4.0/d1/de5/classcv_1_1CascadeClassifier.html
