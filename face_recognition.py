import cv2 as cv
import numpy as np
import os

people=[]
dir=r"C:\Users\jhans\OneDrive\Desktop\face_recognition_system\images_face"
for person in os.listdir(dir):
    people.append(person)

#training
harrcascade=cv.CascadeClassifier("harr_cas.xml")
face_recognition=cv.face.LBPHFaceRecognizer_create()
face_recognition.read("face_trained.yml")

#vedio capturing
video=cv.VideoCapture(0)
while True:
    isTrue,frame=video.read()
    #detect
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_rect=harrcascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in face_rect:
        face_roi=cv.resize(gray[y:y+h,x:x+w],(200,200))
        label,confidence=face_recognition.predict(face_roi)
        if confidence<90:
            cv.putText(frame,str(people[label])+str(confidence),(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.5,(0,225,0),1)
        else:
            cv.putText(frame,"unknown",(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.5,(0,225,0),1)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),2)
        

    cv.imshow("vedio",frame)
    if cv.waitKey(20) & 0xFF==ord("d"):
        break

video.release()
cv.destroyAllWindows()
