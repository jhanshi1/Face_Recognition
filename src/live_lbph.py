import cv2 as cv
from src.detector import FaceDetector
from src.trainer import LBPHTrainer
from src.utils import load_config

def run_lbph_live():
    config = load_config()
    trainer = LBPHTrainer(data_dir=config["data_dir"],cascade_path=config["cascade_path"],face_size=config["face_size"])
    model, people = trainer.train()
    threshold = config.get("lbph_threshold", 146)
    detector = FaceDetector(config["cascade_path"])
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera error")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detector.detect(gray)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv.resize(roi, config["face_size"])
            roi = cv.equalizeHist(roi)
            label, conf = model.predict(roi)
            if conf < threshold:
                name = people[label]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, name, (x, y-10),cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv.imshow("LBPH Live", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
