import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import defaultdict

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16

DATASET_PATH = "dataset/field/test"
MODEL_PATH = "multimodal_model.pth"   # change to image_only_model.pth if needed
USE_ENV = True   # False for image-only model

CLASS_NAMES = [
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Yellow_Leaf_Curl_Virus"
]

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
test_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODELS ----------------
class ImageOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, 5)

    def forward(self, x):
        return self.model(x)

class MultiModalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.env_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, img, env):
        img_feat = self.cnn(img)
        env_feat = self.env_net(env)
        fused = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(fused)

# ---------------- LOAD MODEL ----------------
if USE_ENV:
    model = MultiModalNet().to(DEVICE)
else:
    model = ImageOnlyNet().to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- EVALUATION ----------------
correct = 0
total = 0
class_correct = defaultdict(int)
class_total = defaultdict(int)

# dummy environment input (same used during training)
env_input = torch.tensor([[25, 70, 5]], dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if USE_ENV:
            env = env_input.repeat(images.size(0), 1)
            outputs = model(images, env)
        else:
            outputs = model(images)

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            class_correct[labels[i].item()] += (preds[i] == labels[i]).item()
            class_total[labels[i].item()] += 1

# ---------------- PRINT RESULTS ----------------
print("\n========== MODEL ACCURACY REPORT ==========\n")

print(f"Total Test Images : {total}")
print(f"Overall Accuracy  : {100 * correct / total:.2f}%\n")

print("Class-wise Accuracy:")
print("-" * 40)
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if class_total[cls_idx] > 0:
        acc = 100 * class_correct[cls_idx] / class_total[cls_idx]
        print(f"{cls_name:25} : {acc:.2f}%")

print("\n===========================================\n")