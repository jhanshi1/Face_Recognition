import cv2 as cv
import numpy as np
from keras_facenet import FaceNet

class FaceNetEmbedder:
    def __init__(self):
        self.model = FaceNet()

    def get_embedding(self, face_img):
        face_img = cv.resize(face_img, (160, 160))
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)

        embedding = self.model.embeddings(face_img)
        return embedding[0]
