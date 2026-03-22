
import os
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/lab/train"
ENV_CSV = "env_data.csv"

NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS_STAGE1 = 3
EPOCHS_STAGE2 = 3
EPOCHS_STAGE3 = 4


# =========================================================
# 1. Custom Dataset → Returns (image, label, filepath)
# =========================================================
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


# =========================================================
# 2. Load + Normalize Environmental Data
# =========================================================
env_df = pd.read_csv(ENV_CSV)
env_df["filename"] = env_df["path"].apply(lambda x: os.path.basename(str(x)))

for col in ["temperature", "humidity", "rainfall"]:
    env_df[col] = (env_df[col] - env_df[col].min()) / (env_df[col].max() - env_df[col].min())

env_dict = {
    row["filename"]: torch.tensor([row["temperature"], row["humidity"], row["rainfall"]], dtype=torch.float32)
    for _, row in env_df.iterrows()
}

DEFAULT_ENV = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


# =========================================================
# 3. Heavy Training Augmentation + Normalization
# =========================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = ImageFolderWithPaths(TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# =========================================================
# 4. Multimodal Model (EfficientNet + Env MLP)
# =========================================================
class MultimodalEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.cnn.classifier[1] = nn.Identity()  # 1280 dims

        self.env_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1280 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, env):
        img_feat = self.cnn(img)
        env_feat = self.env_net(env)
        fused = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(fused)


model = MultimodalEfficientNet(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()


# =========================================================
# Helper: Freeze / Unfreeze Layers
# =========================================================
def set_trainable(module, flag=True):
    for p in module.parameters():
        p.requires_grad = flag


# =========================================================
# STAGE 1 — Train Image Classifier Only
# =========================================================
print("\n==============================")
print(" STAGE 1: Train Image Head Only")
print("==============================")

set_trainable(model.cnn, False)
set_trainable(model.env_net, False)
set_trainable(model.classifier, True)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(EPOCHS_STAGE1):
    model.train()
    correct, total = 0, 0

    for imgs, labels, paths in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        dummy_env = torch.zeros((len(imgs), 3)).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs, dummy_env)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    print(f"Stage 1 | Epoch {epoch+1} | Acc: {correct/total:.4f}")


# =========================================================
# STAGE 2 — Train Env MLP + Fusion (Freeze CNN)
# =========================================================
print("\n==============================")
print(" STAGE 2: Train Env + Fusion")
print("==============================")


set_trainable(model.cnn, False)
set_trainable(model.env_net, True)
set_trainable(model.classifier, True)

optimizer = torch.optim.Adam([
    {"params": model.env_net.parameters(), "lr": 3e-3},
    {"params": model.classifier.parameters(), "lr": 1e-3},
])

for epoch in range(EPOCHS_STAGE2):
    model.train()
    correct, total = 0, 0

    for imgs, labels, paths in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        env_batch = []
        for p in paths:
            fname = os.path.basename(p)
            env_batch.append(env_dict.get(fname, DEFAULT_ENV))
        env_batch = torch.stack(env_batch).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs, env_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    print(f"Stage 2 | Epoch {epoch+1} | Acc: {correct/total:.4f}")


# =========================================================
# STAGE 3 — Fine-Tune the Entire Model
# =========================================================
print("\n==============================")
print(" STAGE 3: Fine-Tune Everything")
print("==============================")

set_trainable(model.cnn, True)
set_trainable(model.env_net, True)
set_trainable(model.classifier, True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCH_STAGE3 := EPOCHS_STAGE3):
    model.train()
    correct, total = 0, 0

    for imgs, labels, paths in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        env_batch = []
        for p in paths:
            fname = os.path.basename(p)
            env_batch.append(env_dict.get(fname, DEFAULT_ENV))
        env_batch = torch.stack(env_batch).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs, env_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    print(f"Stage 3 | Epoch {epoch+1} | Acc: {correct/total:.4f}")

torch.save(model.state_dict(), "efficientnet_multimodal_best.pth")
print("\nMultimodal Training Completed Successfully!")