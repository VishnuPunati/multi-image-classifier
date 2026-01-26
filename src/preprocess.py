import os
import random
import shutil
from torchvision.datasets import Caltech101
from PIL import Image

DATA_DIR = "data"
RAW_DIR = "data_raw"
TRAIN_RATIO = 0.8
NUM_CLASSES = 10
SEED = 42

def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def main():
    random.seed(SEED)

    # Clean old data
    clean_dir(DATA_DIR)

    # Download dataset
    dataset = Caltech101(root=RAW_DIR, download=True)

    # Select first N classes
    classes = dataset.categories[:NUM_CLASSES]

    # Create directory structure
    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    # Split data
    for cls in classes:
        indices = [i for i, y in enumerate(dataset.y)
                   if dataset.categories[y] == cls]
        random.shuffle(indices)

        split_idx = int(len(indices) * TRAIN_RATIO)

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        for i in train_idx:
            img, _ = dataset[i]
            img = img.convert("RGB")  # normalize format
            img.save(os.path.join(DATA_DIR, "train", cls, f"{i}.jpg"))

        for i in val_idx:
            img, _ = dataset[i]
            img = img.convert("RGB")  # normalize format
            img.save(os.path.join(DATA_DIR, "val", cls, f"{i}.jpg"))

    print("✅ Dataset preprocessing complete.")
    print(f"Train/Val data saved in '{DATA_DIR}/' directory")

if __name__ == "__main__":
    main()
