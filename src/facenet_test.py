import os
import cv2 as cv
import numpy as np
from numpy.linalg import norm
from src.detector import FaceDetector
from src.facenet_embedder import FaceNetEmbedder

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))

def test_facenet(data_dir, cascade_path, centroids, threshold):
    detector = FaceDetector(cascade_path)
    embedder = FaceNetEmbedder()
    correct = 0
    total = 0
    test_dir = os.path.join(data_dir, "test")
    for person in os.listdir(test_dir):
        person_path = os.path.join(test_dir, person)
        for img in os.listdir(person_path):
            image = cv.imread(os.path.join(person_path, img))
            if image is None:
                continue
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            faces = detector.detect(gray)
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                emb = embedder.get_embedding(face)
                dists = {p: cosine_distance(emb, c) for p, c in centroids.items()}
                pred_person = min(dists, key=dists.get)
                if dists[pred_person] < threshold and pred_person == person:
                    correct += 1
                total += 1
    print(f"FaceNet Test Accuracy: {correct / total:.2f}")
