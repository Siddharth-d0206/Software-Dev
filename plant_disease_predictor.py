import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# === CONFIG ===
DATA_DIR = "C:/Users/sidha/OneDrive/Desktop/plant_disease_dataset"  # Replace with your plant dataset path
BATCH_SIZE = 16
IMAGE_SIZE = 128
EPOCHS = 30
MODEL_PATH = "plant_disease_cnn.pth"

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # If using RGB, this could be [0.5, 0.5, 0.5]
])

# === DATASETS & LOADERS ===
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
CLASSES = dataset.classes  # Save for prediction use

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === CNN MODEL ===
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# === TRAINING SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantCNN(num_classes=len(CLASSES)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# === TRAINING LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # === VALIDATION ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Val Acc: {acc:.2f}")

# === SAVE MODEL + CLASS LABELS ===
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": CLASSES
}, MODEL_PATH)

print(f"✅ Model trained and saved to {MODEL_PATH}")
