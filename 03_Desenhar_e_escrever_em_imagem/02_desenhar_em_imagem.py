#Usar
#python 02_desenhar_em_imagem.py

import cv2
import numpy as np

# Criar imagem com pixels pretos
imagem = np.zeros((512,512,3), np.uint8)

desenhando = False # True se pressionar botao do mouse
retangulo = True # True para desenhar retangulo. Pressionar 'm' para trocar
ix,iy = -1,-1

# mouse callback function
#evento recebe o que foi detectado
#x e y as coordenadas do evento
def desenhar(evento,x,y,flags,param):
    global ix,iy,desenhando,retangulo
    #print('evento: {}'.format(evento))
    #print('flags: {}'.format(flags))
    #print('param: {}'.format(param))
    #Se botao mouse foi pressionado
    if evento == cv2.EVENT_LBUTTONDOWN:
        desenhando = True
        ix,iy = x,y

    elif evento == cv2.EVENT_MOUSEMOVE:
        if desenhando == True:
            if retangulo == True:
                cv2.rectangle(imagem,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(imagem,(x,y),5,(0,0,255),-1)
    #Se botao mouse deixou de ser pressionado
    elif evento == cv2.EVENT_LBUTTONUP:
        desenhando = False
        if retangulo == True:
            cv2.rectangle(imagem,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv2.circle(imagem,(x,y),5,(0,0,255),-1)

cv2.namedWindow('Imagem')
#Ativar a funcao de monitorar eventos do mouse
#Nome da janela no primeiro parametro
#'desenhar' contem o que deve ser monitorado
cv2.setMouseCallback('Imagem',desenhar)

#Listar todos os eventos possiveis
#eventos = [i for i in dir(cv2) if 'EVENT' in i]
#print( eventos )

while(1):
    cv2.imshow('Imagem',imagem)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        retangulo = not retangulo
    elif k == ord('s'):
        cv2.imwrite('desenho.png', imagem)
    elif k == ord('q'):
        break
cv2.destroyAllWindows()

#Referencia
#https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
