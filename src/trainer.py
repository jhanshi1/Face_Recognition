import cv2 as cv
import os
import numpy as np
from src.detector import FaceDetector

class LBPHTrainer:
    def __init__(self, data_dir, cascade_path, face_size):
        self.data_dir = data_dir
        self.face_size = tuple(face_size)
        self.detector = FaceDetector(cascade_path)

    def load_data(self, split):
        features = []
        labels = []
        people = sorted(os.listdir(os.path.join(self.data_dir, split)))

        for label, person in enumerate(people):
            #path
            person_path = os.path.join(self.data_dir, split, person)
            if not os.path.isdir(person_path):
                continue
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path, img)
                image = cv.imread(img_path)
                if image is None:
                    continue
                #converting into gray scale
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                #detect faces
                faces = self.detector.detect(gray)
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    roi = cv.resize(roi, self.face_size)
                    roi = cv.equalizeHist(roi)
                    features.append(roi)
                    labels.append(label)
        return features, np.array(labels), people

    def train(self):
        features, labels, people = self.load_data("train")
        recognizer = cv.face.LBPHFaceRecognizer_create(radius=2,neighbors=16,grid_x=8,grid_y=8)
        recognizer.train(features, labels)
        return recognizer, people
