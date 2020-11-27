#Usar
#python carregar_imagens.py

import cv2
import os
import glob

path = '/home/edee/EdeE/OpenCV/01_Carregar_Midia/Midias/'
for infile in glob.glob(os.path.join(path, '*.jpg')):
    print(infile)
    imagem = cv2.imread(infile)
    cv2.imshow('Imagem', imagem)
    cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
