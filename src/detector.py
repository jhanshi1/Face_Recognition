import cv2 as cv
class FaceDetector:
    def __init__(self, cascade_path):
        #harr_cascade
        self.cascade = cv.CascadeClassifier(cascade_path)

    def detect(self, gray_frame):
        return self.cascade.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=4)
