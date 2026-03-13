import os
import json
import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

MODEL_PATH = "model/image_classifier.pth"
RESULTS_PATH = "results/metrics.json"
BATCH_SIZE = 16

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["classes"]

    weights = MobileNet_V2_Weights.DEFAULT
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    val_ds = datasets.ImageFolder("data/val", transform=val_tfms)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("metrics.json generated successfully")

if __name__ == "__main__":
    main()
