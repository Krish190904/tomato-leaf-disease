import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset

# -------- SETTINGS --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

# -------- TRANSFORMS --------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------- DATA PATHS --------
LAB_TRAIN = "dataset/lab/train"
FIELD_TRAIN = "dataset/field/train"
FIELD_TEST = "dataset/field/test"

# -------- DATASETS --------
lab_train_ds = datasets.ImageFolder(LAB_TRAIN, transform=train_transform)
field_train_ds = datasets.ImageFolder(FIELD_TRAIN, transform=train_transform)
train_ds = ConcatDataset([lab_train_ds, field_train_ds])

test_ds = datasets.ImageFolder(FIELD_TEST, transform=test_transform)

# -------- LOADERS --------
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -------- MODEL --------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(lab_train_ds.classes))
model = model.to(DEVICE)

# -------- TRAINING SETUP --------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------- TRAIN LOOP --------
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Accuracy: {acc:.2f}%")

# -------- TEST LOOP --------
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"\nField Test Accuracy: {test_acc:.2f}%")
torch.save(model.state_dict(), "image_only_model.pth")