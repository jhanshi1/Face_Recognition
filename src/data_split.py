import random
import shutil
from pathlib import Path

def split_dataset(source_dir,output_dir,train_ratio=0.7,val_ratio=0.15,seed=42):
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for person_dir in source_dir.iterdir():
        if not person_dir.is_dir():
            continue
        images = list(person_dir.glob("*"))
        if len(images) < 5:
            print(f"Skipping {person_dir.name}, not enough images")
            continue
        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        for split_name, split_imgs in zip(["train", "val", "test"],[train_imgs, val_imgs, test_imgs]):
            split_person_dir = output_dir / split_name / person_dir.name
            split_person_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy(img, split_person_dir / img.name)
        print(f"{person_dir.name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
if __name__ == "__main__":
    split_dataset(source_dir="data/images_face",output_dir="data")
