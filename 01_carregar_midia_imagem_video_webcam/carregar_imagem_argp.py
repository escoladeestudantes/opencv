#Usar
#python carregar_imagem_argp.py --image FabioBrazza.jpg
#python carregar_imagem_argp.py --image ../GabrielOPensador.jpg
#python carregar_imagem_argp.py --image ../Midias/MCFunkero.jpg
#python carregar_imagem_argp.py --image /home/edee/EdeE/OpenCV/01_Carregar_Midia/Midias/MCFunkero.jpg

#Biblioteca
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

#Carregar imagem
print(args)
image = cv2.imread(args["image"])
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


