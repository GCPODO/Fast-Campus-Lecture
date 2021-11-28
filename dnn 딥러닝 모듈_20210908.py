#OpenCV dnn 딥러닝 모듈

#Deep Neural Network
#Caffe Framework

#1. File Upload
#model_name = 'res10_300x300_ssd_iter_14000.caffemodel'
#prototxt_name = 'deploy.prototxt.txt'
#file_name = 'obama_01.jpeg, obama_02.jpeg, obama_03.jpg'

from.google.colab import files
files.upload()

!ls -al

#2. Load File

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
