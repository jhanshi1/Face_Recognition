import numpy as np
from src.trainer import LBPHTrainer

def validate_lbph(config):
    trainer = LBPHTrainer(
        data_dir=config["data_dir"],
        cascade_path=config["cascade_path"],
        face_size=config["face_size"]
    )

    model, people = trainer.train()
    features, labels, _ = trainer.load_data("val")

    confidences = []

    for img, true_label in zip(features, labels):
        pred_label, confidence = model.predict(img)
        confidences.append(confidence)

    print("Validation confidence stats:")
    print(f"Min: {min(confidences):.2f}")
    print(f"Max: {max(confidences):.2f}")
    print(f"Mean: {np.mean(confidences):.2f}")

    return np.mean(confidences) + np.std(confidences)
