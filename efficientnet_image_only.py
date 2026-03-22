import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "dataset/lab/train"
TEST_DIR = "dataset/field/test"

NUM_CLASSES = 5
BATCH_SIZE = 16

# =========================
# AUGMENTATION (KEEP SAME)
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomRotation(25),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATA
# =========================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_ds = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print("Classes:", train_ds.classes)
print("Total training images:", len(train_ds))

# =========================
# MODEL
# =========================
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# =====================================================
# STAGE 1: TRAIN CLASSIFIER ONLY
# =====================================================
for param in model.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

print("\n========== STAGE 1: Training Classifier ==========")

for epoch in range(5):
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
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Stage 1 | Epoch {epoch+1} | Train Acc: {correct/total:.4f}")

# =====================================================
# STAGE 2: FINE-TUNE ENTIRE MODEL
# =====================================================
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

print("\n========== STAGE 2: Fine-Tuning ==========")

for epoch in range(15):
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
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Stage 2 | Epoch {epoch+1} | Train Acc: {correct/total:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "efficientnet_image_only_improved.pth")

print("\nTraining completed and model saved!")