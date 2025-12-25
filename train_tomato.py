import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Using Apple Silicon GPU (MPS) if available
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", DEVICE)

DATA_DIR = "data"
NUM_CLASSES = 11
BATCH_SIZE = 32
EPOCHS = 10  # Faster training first round
LR = 1e-4
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 160  # Reduced resolution => fast

def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    valid_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=valid_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_dl, valid_dl, train_ds.classes


def train():
    os.makedirs("models", exist_ok=True)
    train_dl, valid_dl, classes = get_dataloaders()

    # MobileNetV2 → Fast + Good Accuracy
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in valid_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Valid Acc: {acc:.4f}")

        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "classes": classes}, MODEL_PATH)
            print("🔥 Best model saved ✔")

    print("\nTraining Finished 🎯")


if __name__ == "__main__":
    train()