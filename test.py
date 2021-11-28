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