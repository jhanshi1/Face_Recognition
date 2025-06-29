import cv2 as cv
import os
import numpy as np

#files import
people=[]
dir=r"C:\Users\jhans\OneDrive\Desktop\face_recognition_system\images_face"
for person in os.listdir(dir):
    people.append(person)

harrcascade=cv.CascadeClassifier("harr_cas.xml")
features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            face_rect=harrcascade.detectMultiScale(gray,1.1,4)
            for (x,y,w,h) in face_rect:
                face_roi=cv.resize(gray[y:y+h,x:x+w],(200,200))
                labels.append(label)
                features.append(face_roi)
create_train()
# features=np.array(features,dtype="object")
labels=np.array(labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save("face_trained.yml")
        
