# 자율주행과 딥러닝

# 자동차 번호판 인식 예제

# haar-cascade 를 사용하여 자동차 번호판 인식

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

print(classes)

from google.colab import files
files.upload()

img = cv2.imread('car1.jpg')
height, width, channels = img.shape

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
       if class_ids[1] == 2:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(i, class_ids[i], label)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), font, 0.5, (0, 255, 0), 1)

cv2_imshow(img)

# 3. Plate

# 'haarascade_russian_plate_number.xml'
from google.colab import files
files.upload()

!ls -al

plate_cascade_name = 'haarcascade_russian_plate_number.xml'

plate_cascade = cv2.CascadeClassifier()
#-- 1. Load the cascades
if not plate_cascade.load(cv2.samples.findFile(plate_cascade_name)):
    print('--(!)'Error loading face cascade')
    exit(0)

img = cv2.imread('russiacar.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

height, width, channels = img.shape

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
       if class_ids[1] == 2:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(i, class_ids[i], label)
            color = colors[i]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(img, label, (x, y + 30), font, 0.5, (0, 255, 0), 1)
            carROI = gray[y:y+h, x:x+w]
            cv2_imshow(carROI)
            plates = plate_cascade.detectMultiScale(carROI)
            print(plates)
            for (x2, y2, w2, h2) in plates:
                cv2.rectangle(img, (x+x2, y+y2), (x+x2 + w2, y+y2 + h2), (0, 255, 0), 2)


cv2_imshow(img)

# 차선인식 프로젝트

# Edge Detection / Region Of Interest / Lane Detection

# Hough Transform - 임의의 점들은 찾음. 그걸 통해 선을 식별

from google.colab import files
files.upload

!ls -al

import cv2
import numpy as np
import math
from google.colab.patches import cv2_imshow

file_name = 'car3.jpeg'
frame = cv2.imread(file_name)
cv2_imshow(frame)
height, width, channels = frame.shape
print(height, width, channels) # 540 960 3

# Covert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
cv2_imshow(gray)
height, width = gray.shape
print(height, width) # 540 960

# GaussianBlur for reducing noise -> 노이즈를 줄이기 위함
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2_imshow(blur)

canny = cv2.Canny(blur, 40, 130) # 임계값 설정에 따라 더 엄격하게 가능
cv2_imshow(canny)

# 2. Region of Interest

mask = np.zeros((height, width), dtype = 'unint8')
cv2_imshow(mask)

poly_height = int(0.60 * height)
poly_left = int(0.47 * width)
poly_right = int(0.53 * width)
polygons = np.array([[(0, height), (poly_left, poly_height), (poly_right, poly_height), (width, height)]])
cv2.fillPoly(mask, polygons, 255)
cv2_imshow(mask)

# Bitwise operation between poly and mask
masked = cv2.bitwise_and(canny, mask)
cv2_imshow(masked)

# 3. Lane Detection

lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 20, np.array([]), 20, 10)

image_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
if lines is not None:
        for line in lines:
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2_imshow(image_rgb)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
cv2_imshow(frame)

# kaggle.com -> 차선 인식에 관련된 여러 데이터 소스 제공

# 1. File Load

from google.colab import files
files.upload()

import cv2
import numpy as np
import IPython
from google.colab.patches import cv2_imshow

file_name = 'test_video.mp4'

def detectAndDisplay(frame):
    IPython.display.clear_output(wait=True)
    # Covert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    # GaussianBlur for reducing noise -> 노이즈를 줄이기 위함
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    canny = cv2.Canny(blur, 40, 130)  # 임계값 설정에 따라 더 엄격하게 가능
    mask = np.zeros((height, width), dtype='unint8')
    poly_height = int(0.60 * height)
    poly_left = int(0.47 * width)
    poly_right = int(0.53 * width)
    polygons = np.array([[(0, height), (poly_left, poly_height), (poly_right, poly_height), (width, height)]])
    cv2.fillPoly(mask, polygons, 255)
    # Bitwise operation between poly and mask
    masked = cv2.bitwise_and(canny, mask)
    lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 20, np.array([]), 20, 10)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    cv2_imshow(frame)

#-- 2. Read the video stream
cap = cv2.VideoCapture(file_name)

if not cap.isOpened:
    print('--(!)Error opening vido capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)

# 도로 표지판 인식 프로젝트 / 도로 차량 추적 프로젝트

# YOLO 학습 프로젝트 소개












