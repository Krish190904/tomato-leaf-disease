import os
import csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

# ---------------- LOAD ENV DATA ----------------
env_dict = {}

with open("env_data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        env_dict[row["path"]] = torch.tensor(
            [
                float(row["temperature"]),
                float(row["humidity"]),
                float(row["rainfall"]),
            ],
            dtype=torch.float32,
        )

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

# ---------------- CUSTOM DATASET ----------------
class MultiModalDataset(Dataset):
    def __init__(self, root_dir):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        img_path = self.dataset.samples[idx][0]

        # Convert absolute path → relative path used in CSV
        rel_path = img_path.split("dataset/")[-1].replace("\\", "/")

        if rel_path not in env_dict:
            raise KeyError(f"Missing env data for: {rel_path}")

        env = env_dict[rel_path]
        return image, env, label


# ---------------- DATASETS ----------------
lab_train = MultiModalDataset("dataset/lab/train")
field_train = MultiModalDataset("dataset/field/train")
train_dataset = ConcatDataset([lab_train, field_train])

test_dataset = MultiModalDataset("dataset/field/test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
class MultiModalNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Image branch
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        # Environment branch
        self.env_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, env):
        img_feat = self.cnn(img)
        env_feat = self.env_net(env)
        fused = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(fused)


model = MultiModalNet(num_classes=5).to(DEVICE)

# ---------------- TRAINING SETUP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- TRAIN ----------------
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for imgs, envs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        envs = envs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs, envs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Accuracy: {acc:.2f}%")

# ---------------- TEST ----------------
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for imgs, envs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        envs = envs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs, envs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nField Test Accuracy (Image + Env): {100 * correct / total:.2f}%")
torch.save(model.state_dict(), "multimodal_model.pth")
