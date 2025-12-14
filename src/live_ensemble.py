import cv2 as cv
from src.detector import FaceDetector
from src.trainer import LBPHTrainer
from src.facenet_embedder import FaceNetEmbedder
from src.facenet_train import build_embeddings
from src.facenet_validate import cosine_distance
from src.utils import load_config

def run_ensemble_live():
    config = load_config()
    # LBPH
    trainer = LBPHTrainer(data_dir=config["data_dir"],cascade_path=config["cascade_path"],face_size=config["face_size"])
    lbph_model, people = trainer.train()
    lbph_th = config.get("lbph_threshold", 146)
    # FaceNet
    embedder = FaceNetEmbedder()
    centroids = build_embeddings(config["data_dir"], config["cascade_path"])
    facenet_th = config.get("facenet_threshold", 0.55)
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
            # FaceNet decision
            face = frame[y:y+h, x:x+w]
            emb = embedder.get_embedding(face)
            dists = {p: cosine_distance(emb, c) for p, c in centroids.items()}
            best = min(dists, key=dists.get)
            if dists[best] > facenet_th:
                label = "Unknown"
                color = (0, 0, 255)
            else:
                # FaceNet accepted already (dists[best] <= facenet_th)
                # Optional LBPH sanity check 
                roi = gray[y:y+h, x:x+w]
                roi = cv.resize(roi, config["face_size"])
                roi = cv.equalizeHist(roi)
                lbph_label, conf = lbph_model.predict(roi)
                label = best
                color = (0, 255, 0)
                # If LBPH strongly disagrees, downgrade confidence
                if conf > lbph_th * 1.3:
                    label = f"{best}?"
                    color = (0, 255, 255)
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, label, (x, y-10),cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv.imshow("Ensemble Live", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
