
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matploplib.pyplot as plt

print("OpenCV version: ",cv2.__version__)

img = cv2.imread('car1.jpg')


cv2_imshow(img)

print("width: {} pixels".format(img.shape[1]))
print("width: {} pixels".format(img.shape[0]))
#shape[0]은 height shape[1]은 width
print("channels: {}".format(img.shape[2])) #channel 은 shape[3]

(height, width) = img.shape[:2]
center = (height //2, width //2)

print(height, width,center)


X = 0  #@param {type:"slider",min:0,max:1280,step:1}
Y = 0  #@param {type:"slider",min:0,max:1280,step:1}
SIZE = 0  #@param {type:"slider",min:0,max:200,step:1}
#코랩에서 슬라이더가 나와서 값을 조절할 수 있게 됨.

(b,g,r) = img[X,Y]
print("Pixel at ({},{}) - Red: {}, Green: {}, Blue{}".format(X,Y,r,g,b))


#Crop cordination = image[y: y+h, x:x+w]
croped = img[y: y+SIZE, x:x+SIZE] = (0.0.225)
cv2_imshow(croped)

cv2.rectangle(img,(X+SIZE*2, Y),(X+SIZE*3,Y+SIZE),(0,255,0),5)

radius = int(SIZE/2)
cv2.circle(img,(X+SIZE*4,Y+radius),radius,(255,255,0),-1)

cv2.line(img, (X+SIZE*5,Y),(X+SIZE*6,Y+SIZE),(0.255,255),5)

cv2.putText(img,'creApple',(X+SIZE*7,Y+SIZE),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))

#왼쪽 위에서부터, x가 오른쪽 y 가 아래
#하나씩 해보면서 그림이 어디에 그려지는지, 좌표가 어떻게 생겼는지 보기


#저장

cv2.imwrite('car-copy.jpg',img)

#Moved down: +,up:- and right: +, left -

move = np.float32([[1,0,100],[0,1,100]])
moved = cv2.watpAffine(img,move,(width, height))
cv2_imshow(moved)


#사진 돌리기
rotate = cv2.getRatationMatrix2D(center,90,1.0)
ratated = cv2. warpAffine(img,roatate,(width,height))
cv2_imshow(rotated)



ratio = SIZE / WIDTH
dimention = (SIZE, int(height * ratio))

resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)#resize할 때 어떻게 찌그러트릴지


#사진 뒤집기
flipped = cv2.flip(img, 1)
cv2_imshow(flipped)





*OPenCV _ Mask

#백그라운드는 np에서 정의
background = np.full((height, width ,3 ),255,np.uint8)

plt.figure(figsize(15,15))
plt.imshow(background)

#plt는 matplotlib의 명령임.
cv2.imwrite('plt-copy.jpg',background)


#Mask

#White background = np.full((height, width ,3 ),255,np.uint8)

mask = np.zeros(img.shape[:2],dtype='uint8')
#0들이 꽉 채워짐.
cv2.circle(mask, center, int(height/2),(255,255,255),-1)
#0들이 채워진 화면에 동그란 원을 올림

cv2_imshow(mask)   #검은 배경에 동그란 원이 들어있음.


masked = cv2.bitwise_and(img, img, mask = mask)
cv2_imshow(masked)  #동그란 부분에만 표시되는 사진이 나옴. point : 흰 부분만 볼 수 있다.



#Fileter

zeros = np.zeros(img.shape[:2], np.uint=8 )

cv2_imshow(zeros) #그냥 까만색 화면

(Blue, Green, Red) = cv2.split(img)
cv2_imshow(Blue)

cv2_imshow(cv2.merge(Blue, zeros, zeros)) #파란색만 강조됨.

cv2_imshow(cv2.merge(zeros, Green, zeros)) #초록색만

cv2_imshow(cv2.merge(zeros, zeros, Red)) # 빨강색만


#+HSV 개념 : Hue 색상   Saturation 채도  Value 명도

hsv = cv2.cvtColor(img,cv2.COLOR BGR2HSV)  # 컴퓨터가 보기 좋게 만들어주는 것. 컴퓨터가 공부할 양이 적어지는 것.

# lab filter

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow(lab)


ret, thresh = cv2.threshold(gray, 127, 255, 0)

#127을 기준으로 0 이면 0 1이면 1 확실히 구분지어 주는 것.

contounrs, hieracy = cv2.findCountours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

len(contours) #윤곽선이 약 1000개 넘음

contours = sorted(contours, key = cv2.contourArea, reverse = Trus) [:10]

#길이순으로 열개만 짜름
back = np.zeros((height,width,3),np.uint8)

for i in range(len(contours)):
    cv2.drawContours(back,contours, i,(0,255,255))
cv2_imshow(back)

cv2_imshow(thresh)








*Haar-cascade Detection

#사진으로 인식하
import cv2

imgfile = 'C:/Users/user/Desktop/KUO/try/obama_02.jpg'
face_cascade_name = 'C:/Users/user/Desktop/KUO/try/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'C:/Users/user/Desktop/KUO/try/haarcascade_eye.xml'

img = cv2.imread(imgfile)

face_cascade = cv2.CascadeClassifier(face_cascade_name)
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_name)



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)


faces = face_cascade.detectMultiScale(gray) #학습데이터를 이용해서 face를 인식 함.

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    faceROI = img[y:y+h,x:x+h]
    eyes = eyes_cascade.detectMultiScale(faceROI)
    for (x2, y2, w2, h2) in eyes:
        eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        radius = int(round((w2 + h2) * 0.25))
        img = cv2.circle(img, eye_center, radius, (0, 255, 255), 3)

img = cv2.resize(img, (400, 400))
cv2.imshow("OBAMA",img)
cv2.waitKey()
cv2.destroyAllWindows()



*Haar-cascade Detection으로 동영상 파일 인식하기.
-> 동영상은 어차피 픽셀의 연속이므로, 사진과 동일

#동영상으로 인식하
import cv2



face_cascade_name = 'C:/Users/user/Desktop/KUO/try/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'C:/Users/user/Desktop/KUO/try/haarcascade_eye.xml'



#-- 1. Load the cascades

face_cascade = cv2.CascadeClassifier(face_cascade_name)
eyes_cascade = cv2.CascadeClassifier(eyes_cascade_name)




def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #Detect face
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        faceROI = frame_gray[y:y+h, x:x+w]
        # Detect Eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x+x2+w2//2,y+y2+h2//2)
            radius = int(round((w2+h2)*0.25))
            frame = cv2.circle(frame,eye_center,radius,(255,0,0),3)
    



capture = cv2.VideoCapture("C:/Users/user/Desktop/KUO/try/obama_01.mp4")



while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    detectAndDisplay(frame)
    cv2.imshow("VideoFrame", frame)

cv2.destroyAllWindows()

-Haar-cascade Detection


 

