

import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/lab/train"
NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 10

# ==============================
# DATA TRANSFORMS
# ==============================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomPerspective(0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# DATASET & DATALOADER
# ==============================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ==============================
# MODEL: RESNET18 IMAGE-ONLY
# ==============================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        total += len(labels)
        correct += (preds == labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {correct/total:.4f}")

# ==============================
# SAVE MODEL
# ==============================
torch.save(model.state_dict(), "resnet18_image_only.pth")
print("Saved: resnet18_image_only.pth")
