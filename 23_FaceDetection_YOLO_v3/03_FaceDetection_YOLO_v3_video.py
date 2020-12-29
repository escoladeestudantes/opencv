#Usar
#python 03_FaceDetection_YOLO_v3_video.py
import numpy as np
import cv2

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            color = (0,255,0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, str(round(confidences[i],3)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return image

def make_prediction(net, layer_names, image, confidence, threshold):
    height, width = image.shape[:2]
    
    # Pre-processar a imagem para ela tornar-se blob
    # Passar pelo modelo Yolo
    #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extrair os retangulos, confianca e classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Aplicar Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

# Carregar o modelo e os pesos
net = cv2.dnn.readNetFromDarknet('yolov3-face.cfg', 'yolov3-wider_16000.weights')

use_gpu = 0
if (use_gpu == 1):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture('EduardoMarinho.mp4')
#cap = cv2.VideoCapture(0)

ret=True
while (ret==True):
	ret, frame = cap.read()
	if (ret == True):
		boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, frame, 0.1, 0.3)
		frame = draw_bounding_boxes(frame, boxes, confidences, classIDs, idxs)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
cv2.destroyAllWindows()

