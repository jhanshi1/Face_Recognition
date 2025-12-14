import os
import cv2 as cv
import numpy as np
from numpy.linalg import norm
from src.detector import FaceDetector
from src.facenet_embedder import FaceNetEmbedder

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def validate_facenet(data_dir, cascade_path, centroids):
    detector = FaceDetector(cascade_path)
    embedder = FaceNetEmbedder()
    distances = []
    val_dir = os.path.join(data_dir, "val")
    for person in os.listdir(val_dir):
        person_path = os.path.join(val_dir, person)
        for img in os.listdir(person_path):
            image = cv.imread(os.path.join(person_path, img))
            if image is None:
                continue
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            faces = detector.detect(gray)
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                emb = embedder.get_embedding(face)
                d = cosine_distance(emb, centroids[person])
                distances.append(d)
    print("FaceNet validation distance stats:")
    print("Min:", min(distances))
    print("Max:", max(distances))
    print("Mean:", np.mean(distances))
    return np.mean(distances) + np.std(distances)