#Usar
#python 04_body_hog_video.py

#Biblioteca
import cv2

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Video
#cap = cv2.VideoCapture('Bangkok-31965-Expatsiam.mp4')
cap = cv2.VideoCapture('Istanbul-26920-hulkiokantabak.mp4')

ret=True
while(ret==True):
	ret, frame = cap.read()
	#ret==False se acabou video
	if (ret==True):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		(regions, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.15)
		for (x,y,w,h) in regions:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.imshow('Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cap.release() 
cv2.destroyAllWindows()

