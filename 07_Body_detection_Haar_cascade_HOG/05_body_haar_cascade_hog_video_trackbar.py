#Usar
#python 05_body_haar_cascade_hog_video_trackbar.py

#Biblioteca
import cv2

def nothing(x):
	pass

def parametro():
	parametro = cv2.getTrackbarPos("Dim_Scale", "Imagem")
	if parametro < 1050:
		parametro = 1050
	return (parametro / 1000)

fullbody_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')
lowerbody_detector = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
upperbody_detector = cv2.CascadeClassifier('haarcascade_upperbody.xml')
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Imagem
#imagem_original = cv2.imread('pedestrian-zone-347468_640-joergelman-241223.jpg')
#gray = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
#Video
#cap = cv2.VideoCapture('Bangkok-31965-Expatsiam.mp4')
cap = cv2.VideoCapture('Istanbul-26920-hulkiokantabak.mp4')

cv2.namedWindow("Imagem")
cv2.createTrackbar("Dim_Scale", "Imagem", 1100, 1800, nothing)

while(True):
	ret, imagem = cap.read()
	gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	#imagem = imagem_original.copy()
	
	dim_scale = parametro()
	fullbody = fullbody_detector.detectMultiScale(gray, dim_scale, 5)
	lowerbody = lowerbody_detector.detectMultiScale(gray, dim_scale, 5)
	upperbody = upperbody_detector.detectMultiScale(gray, dim_scale, 5)
	(regions, _) = hog.detectMultiScale(imagem, winStride=(4, 4), padding=(4, 4), scale=dim_scale)
	for (x,y,w,h) in fullbody:
		cv2.rectangle(imagem,(x,y),(x+w,y+h),(255,0,0),2)
	for (x,y,w,h) in lowerbody:
		cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,255,0),2)
	for (x,y,w,h) in upperbody:
		cv2.rectangle(imagem,(x,y),(x+w,y+h),(0,0,255),2)
	for (x,y,w,h) in regions:
		cv2.rectangle(imagem,(x,y),(x+w,y+h),(255,255,0),2)

	cv2.imshow('Imagem', imagem)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#cap.release() 
cv2.destroyAllWindows()

