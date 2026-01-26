import os
import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch import nn, optim
from torch.utils.data import DataLoader

MODEL_PATH = os.getenv("MODEL_PATH", "model/image_classifier.pth")
EPOCHS = 2
BATCH_SIZE = 16

def main():
    weights = MobileNet_V2_Weights.DEFAULT
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )


    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_ds = datasets.ImageFolder("data/train", transform=train_tfms)
    val_ds = datasets.ImageFolder("data/val", transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = mobilenet_v2(weights=weights)

    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, len(train_ds.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_ds.classes
                },
                MODEL_PATH
            )

    print("✅ Training finished")

if __name__ == "__main__":
    main()
