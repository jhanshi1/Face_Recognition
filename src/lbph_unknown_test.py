import cv2 as cv
import os
from src.detector import FaceDetector

def test_unknown(config, model, threshold):
    detector = FaceDetector(config["cascade_path"])
    unknown_dir = "data/unknown"
    false_accepts = 0
    total = 0
    for img in os.listdir(unknown_dir):
        image = cv.imread(os.path.join(unknown_dir, img))
        if image is None:
            continue
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = detector.detect(gray)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv.resize(roi, config["face_size"])
            roi = cv.equalizeHist(roi)
            _, confidence = model.predict(roi)
            total += 1
            if confidence < threshold:
                false_accepts += 1
    print(f"False Acceptance Rate: {false_accepts / total:.2f}")
