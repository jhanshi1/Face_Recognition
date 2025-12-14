import cv2 as cv
from src.detector import FaceDetector
from src.facenet_embedder import FaceNetEmbedder
from src.facenet_train import build_embeddings
from src.facenet_validate import cosine_distance
from src.utils import load_config


def run_facenet_live():
    config = load_config()
    detector = FaceDetector(config["cascade_path"])
    embedder = FaceNetEmbedder()
    centroids = build_embeddings(config["data_dir"], config["cascade_path"])
    threshold = config.get("facenet_threshold", 0.55)
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
            face = frame[y:y+h, x:x+w]
            emb = embedder.get_embedding(face)
            dists = {p: cosine_distance(emb, c) for p, c in centroids.items()}
            best = min(dists, key=dists.get)
            dist = dists[best]
            if dist < threshold:
                label = f"{best} ({dist:.2f})"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, label, (x, y-10),cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv.imshow("FaceNet Live", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
