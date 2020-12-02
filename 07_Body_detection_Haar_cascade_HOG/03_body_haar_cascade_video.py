#Usar
#python 03_body_haar_cascade_video.py

#Biblioteca
import cv2

fullbody_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')
lowerbody_detector = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
upperbody_detector = cv2.CascadeClassifier('haarcascade_upperbody.xml')
#Video
#cap = cv2.VideoCapture('Bangkok-31965-Expatsiam.mp4')
cap = cv2.VideoCapture('Istanbul-26920-hulkiokantabak.mp4')
#Frame capturado
ret=True
while(ret==True):
	ret, frame = cap.read()
	#ret==False se acabou video
	if (ret==True):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fullbody = fullbody_detector.detectMultiScale(gray, 1.11, 5)
		lowerbody = lowerbody_detector.detectMultiScale(gray, 1.11, 5)
		upperbody = upperbody_detector.detectMultiScale(gray, 1.11, 5)
		for (x,y,w,h) in fullbody:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		for (x,y,w,h) in lowerbody:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		for (x,y,w,h) in upperbody:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.imshow('Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cap.release() 
cv2.destroyAllWindows()

