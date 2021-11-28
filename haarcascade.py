# OpenCV dnn 딥러닝 모듈

# Deep Neural Network
# Caffe Framework

# 1. File Upload
# model_name = 'res10_300x300_ssd_iter_14000.caffemodel'
# prototxt_name = 'deploy.prototxt.txt'
# file_name = 'obama_01.jpeg, obama_02.jpeg, obama_03.jpg'

from.google.colab import files
files.upload()

!ls -al

# 2. Load File

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

model_name = 'res10_300x300_ssd_iter_14000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
file_name = 'obama_01.jpeg'

min_confidence = 0.3
frame_width = 300
frame_height = 300

frame = cv2.imread(file_name)
(height, width) = frame.shape[:2]
print(height, width)

cv2_imshow(frame)

# 3.DNN Model

model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
model.setInput(blob)
detections = model.forward()
print(detections)
for i in range(0, detections.shpe[2]):
    # extract the confidence (i.e., probability)
    confidence = detections[0, 0, i, 2]
    if confidence > min_confidence:
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype('int')
        print(confidence, startX, startY, endX, endY)
        text = '{:.2f}%'.format(confidence*100)
        y = startY - 10 if startY - 10 > 10else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
# Show the output
cv2_imshow(frame)


#DNN Video

# 1. File upload
# model_name = 'res10_300x300_ssd_iter_14000.caffemodel'
# prototxt_name = 'deploy.prototxt.txt'
# file_name = 'obama_01,mp4'

from.google.colab import files
files.upload()

!ls -al

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import IPython

model_name = 'res10_300x300_ssd_iter_14000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
file_name = 'obama_01.mp4'

min_confidence = 0.3
frame_width = 300
frame_height = 300

# DNN Model

model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
def detectAndDisplay(frame):
    IPython.display.clear_output(wait=True)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detections = model.forward()
    for i in range(0, detections.shpe[2]):
        # extract the confidence (i.e., probability)
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            (height, width) = frame.shape[:2]
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype('int')
            # print(confidence, startX, startY, endX, endY)               #안해줘도 괜찮음
            text = '{:.2f}%'.format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    # Show the output
    cv2_imshow(frame)

#-- 2.Read the video stream
cap = cv2.VideoCapture(file_name)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)


# YOLO란 무엇인가

# YOLO : Real-Time Object Detection
# SSD

# 보통 YOLOv3-416 사용
# Tiny YOLO를 보통 라즈베리파이에서 사용(정확도는 좀 떨어질 수 있음)
# Darknet을 먼저 알아야 함. C/CUDA 기반
# CPU GPU 동시 지원

# darknet/cfg/yolov3.cfg 랑 coco.data
# darknet/data/coco.names
# darknet/weights/

# 1. Mount Drive

from google.colab import drive
drive.mount('/gdrive')

!ls -al '/gdrive/My Drive/darknet'

weight_file = '/gdrive/My Drive/darknet/weights/yolov3.weights'
cfg_file = '/gdrive/My Drive/darknet/cfg/yolov3.cfg'
name_file = '/gdrive/My Drive/darknet/data/coco.names'

# 2. Load File

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

min_confidence = 0.5

# Load Yolo

net = cv2.dnn.readNet(weight_file, cfg_file)

classes = []
with open(name_file, 'r') as f:
    classes = [ line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

from google.colab import files
files.upload()

img = cv2.imread('car1.jpg')
height, width, channels = img.shape
cv2_imshow(img)

# Detecting objects
# https://docs.opencv.org/master/d6/d0f/group_dnn.html
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidneces, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_COMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 0.5, (0, 255, 0), 1)

cv2_imshow(img)


# YOLO 영상 처리

# 1. Mount Drive

from google.colab import drive
drive.mount('/gdrive')

!ls -al '/gdrive/My Drive/darknet'

weight_file = '/gdrive/My Drive/darknet/weights/yolov3.weights'
cfg_file = '/gdrive/My Drive/darknet/cfg/yolov3.cfg'
name_file = '/gdrive/My Drive/darknet/data/coco.names'

file_name = 'obama_01.mp4'

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import IPython

min_confidence = 0.5

# Load Yolo
net = cv2.dnn.readNet(weight_file, cfg_file)

classes = []
with open(name_file, 'r') as f:
    classes = [ line.strip() for line in f.readlines()]
# print(classes) -> 치면 목록이 나옴
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

from google.colab import files
files.upload()

def detectAndDisplay(frame):
    IPython.display.clear_output(wait=True)
    height, width, channels = frame.shpae #YOLO는 상대적인 값인데 절대적인 수치로 변환하는 코드
    # Detecting objects
    # https://docs.opencv.org/master/d6/d0f/group_dnn.html
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # -> opencv 들어가면 항목들이 나옴/416 쓰는것이 가성비가 좋다(YOLO)

    net.setInput(blob) #NET이 분석한다
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:] #score는 다섯번째부터 쭈루룩 있는데, 이 것들중에 가장 가능성이 높은 것을 골라서 표시.
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height) #detection[] 로 쓴 이유는 YOLO가 상대적인 값을 필사하기때문에 사진의 width 와 height를 곱해서 실질적인 값으로 바꿔주는 것.

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidneces, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_COMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label)
        color = colors[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y + 30), font, 0.5, (0, 255, 0), 1)

cv2_imshow(frame)

#-- 2.Read the video stream
cap = cv2.VideoCapture(file_name)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
