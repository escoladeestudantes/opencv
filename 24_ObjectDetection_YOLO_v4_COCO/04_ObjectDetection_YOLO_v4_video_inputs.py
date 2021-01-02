#Usar
#python 04_ObjectDetection_YOLO_v4_video_inputs.py
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


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def make_prediction(net, layer_names, labels, image, confidence, threshold, resize_input):
    height, width = image.shape[:2]
    # Pre-processar a imagem para ela tornar-se blob
    # Passar pelo modelo Yolo
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (resize_input, resize_input), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extrair os retangulos, confianca e classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Aplicar Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

# Objetos que o modelo detecta
labels = open('coco.names').read().strip().split('\n')

# Gerar cores aleatoriamente para cada categoria de objeto
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Carregar o modelo e os pesos
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

use_gpu = 1
if (use_gpu == 1):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture('Bangkok-30945.mp4')
ret=True
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
resize_input_yolov4 = 320 #320, 416, 512, 608
video = cv2.VideoWriter('Bangkok-30945-yolov4-{}.avi'.format(resize_input_yolov4),  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
text = str('YOLOv4_{}x{}'.format(resize_input_yolov4,resize_input_yolov4))
while (ret==True):
    ret, frame = cap.read()
    if (ret == True):
        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, frame, 0.1, 0.3, resize_input_yolov4)
        frame = draw_bounding_boxes(frame, boxes, confidences, classIDs, idxs, colors)
        frame = cv2.putText(frame, text, (int(0.75*frame_width), int(0.97*frame_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        #cv2.imshow("Frame", frame)
        video.write(frame)
cv2.destroyAllWindows()

