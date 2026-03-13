import os
from dotenv import load_dotenv

load_dotenv()

API_PORT = int(os.getenv("API_PORT", 8000))
MODEL_PATH = os.getenv("MODEL_PATH", "model/image_classifier.pth")

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

RESULTS_PATH = "results/metrics.json"
