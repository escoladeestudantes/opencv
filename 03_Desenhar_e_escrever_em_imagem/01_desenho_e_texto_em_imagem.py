#Usar
#python 01_desenho_e_texto_em_imagem.py

#Biblioteca
import cv2

imagem = cv2.imread('HipHop_by_real_napster.jpg')

#Linha
#cv2.line(Imagem, (coluna_origem, linha_origem), (coluna_final, linha_final), (Azul,Verde,Vermelho), espessura)
cv2.line(imagem, (0, 0), (350, 100), (0,0,0), 4)
#Retangulo
#cv2.rectangle(Imagem, (coluna_origem, linha_origem), (coluna_final, linha_final), (Azul,Verde,Vermelho), espessura)
cv2.rectangle(imagem, (5, 25), (350, 100), (0,0,0), 4)
#Parametro da espessura -1 preenche a parte interna
cv2.rectangle(imagem, (5, 125), (40, 280), (0,0,0), -1)
#Circulo
#cv2.circle(Imagem, (coluna_central, linha_central), diametro, (Azul,Verde,Vermelho), espessura)
cv2.circle(imagem, (80,250), 20, (0,0,0), 4)
#Parametro da espessura -1 preenche a parte interna
cv2.circle(imagem, (160,250), 30, (0,0,0), -1)
#Texto
#cv2.putText(imagem, ‘Texto’, (coluna_inicial, linha_inicial), fonte, expessura, cor, tipo de linha)
cv2.putText(imagem, 'RAP', (20,370), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), cv2.LINE_AA)
cv2.imwrite('OpenCV_04.png',imagem)
cv2.imshow("Imagem", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Referencia
#https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html


