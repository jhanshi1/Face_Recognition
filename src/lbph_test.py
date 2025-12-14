from src.trainer import LBPHTrainer

def test_lbph(config, threshold):
    trainer = LBPHTrainer(
        data_dir=config["data_dir"],
        cascade_path=config["cascade_path"],
        face_size=config["face_size"]
    )

    model, people = trainer.train()
    features, labels, _ = trainer.load_data("test")

    correct = 0
    total = len(labels)

    for img, true_label in zip(features, labels):
        pred_label, confidence = model.predict(img)

        if confidence < threshold and pred_label == true_label:
            correct += 1

    accuracy = correct / total
    print(f"LBPH Test Accuracy: {accuracy:.2f}")
