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




capture = cv2.VideoCapture("C:/Users/user/Desktop/KUO/try/obama4.mp4")



while cv2.waitKey(33) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    detectAndDisplay(frame)
    cv2.imshow("VideoFrame", frame)

cv2.destroyAllWindows()