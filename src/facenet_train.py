import os
import cv2 as cv
import numpy as np
from src.detector import FaceDetector
from src.facenet_embedder import FaceNetEmbedder

def build_embeddings(data_dir, cascade_path):
    detector = FaceDetector(cascade_path)
    embedder = FaceNetEmbedder()
    embeddings = {}
    counts = {}
    train_dir = os.path.join(data_dir, "train")
    for person in os.listdir(train_dir):
        person_path = os.path.join(train_dir, person)
        if not os.path.isdir(person_path):
            continue
        for img in os.listdir(person_path):
            image = cv.imread(os.path.join(person_path, img))
            if image is None:
                continue
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            faces = detector.detect(gray)
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                emb = embedder.get_embedding(face)
                embeddings.setdefault(person, np.zeros_like(emb))
                counts[person] = counts.get(person, 0) + 1
                embeddings[person] += emb
    # centroid per person
    for person in embeddings:
        embeddings[person] /= counts[person]
    return embeddings
