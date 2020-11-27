#Usar
#python carregar_imagem.py

#Biblioteca
import cv2
#Pode-se usar ' as novo_nome' (sem aspas simples) para mudar (simplificar) o nome
#Exemplo import cv2 as cv

#Carregar imagem
#----
#variavel = cv2.imread('caminho_para_imagem', flag)
#Flag Padrao: leitura de imagem colorida desconsiderando qualquer transparencia, cv2.IMREAD_COLOR ou o valor 1
# Flag para imagem em tons de cinza: cv2.IMREAD_GRAYSCALE ou 0
# Flag para imagem incluindo o canal alfa: cv2.IMREAD_UNCHANGED ou -1
#----
#imagem = cv2.imread('FabioBrazza.jpg')
#imagem = cv2.imread('FabioBrazza.jpg', 1)
#imagem = cv2.imread('FabioBrazza.jpg', 0)
#imagem = cv2.imread('../GabrielOPensador.jpg')
#imagem = cv2.imread('../Midias/GigantenoMic.jpg')
#imagem = cv2.imread('/home/edee/EdeE/OpenCV/01_Carregar_Midia/Midias/MCFunkero.jpg')
imagem = cv2.imread('Technology.jpg')
#Mostrar imagem
#----
#cv2.imshow('Nome_da_janela',variavel_ou_funcao_de_leitura_cv2.imread...)
#----
cv2.imshow('Imagem',imagem)
#Aguardar tecla ser pressionada ou determinado tempo
cv2.waitKey(0)
#cv2.waitKey(1500)
cv2.destroyAllWindows()

#Referencia
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html


