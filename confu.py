import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = "dataset/field/test"

RESNET_PATH = "C:\\Users\\uroha\\OneDrive\\Desktop\\faarzi\\resnet18_image_only.pth"
EFFNET_PATH = "C:\\Users\\uroha\\OneDrive\\Desktop\\faarzi\\efficientnet_image_only_improved.pth"

BATCH_SIZE = 16

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATA
# =========================
test_ds = ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_ds.classes

print("Classes:", class_names)
print("Total test samples:", len(test_ds))


# =========================
# LOAD MODELS
# =========================
def load_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_effnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(EFFNET_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# =========================
# GET PREDICTIONS
# =========================
def get_predictions(model):
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_pred)


# =========================
# RUN
# =========================
resnet = load_resnet()
effnet = load_effnet()

y_true_r, y_pred_r = get_predictions(resnet)
y_true_e, y_pred_e = get_predictions(effnet)

# =========================
# ACCURACY
# =========================
acc_r = np.mean(y_true_r == y_pred_r)
acc_e = np.mean(y_true_e == y_pred_e)

print(f"\nResNet18 Accuracy:     {acc_r:.4f}")
print(f"EfficientNet Accuracy: {acc_e:.4f}")


# =========================
# CONFUSION MATRICES
# =========================
cm_r = confusion_matrix(y_true_r, y_pred_r)
cm_e = confusion_matrix(y_true_e, y_pred_e)

# =========================
# PLOT BOTH SIDE BY SIDE
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14,6))

# ResNet
im1 = axes[0].imshow(cm_r, cmap="Greens")
axes[0].set_title("ResNet18 Confusion Matrix")
axes[0].set_xticks(range(len(class_names)))
axes[0].set_yticks(range(len(class_names)))
axes[0].set_xticklabels(class_names, rotation=45)
axes[0].set_yticklabels(class_names)

# EfficientNet
im2 = axes[1].imshow(cm_e, cmap="Greens")
axes[1].set_title("EfficientNet Confusion Matrix")
axes[1].set_xticks(range(len(class_names)))
axes[1].set_yticks(range(len(class_names)))
axes[1].set_xticklabels(class_names, rotation=45)
axes[1].set_yticklabels(class_names)

# Add numbers inside cells
for i in range(cm_r.shape[0]):
    for j in range(cm_r.shape[1]):
        axes[0].text(j, i, cm_r[i, j], ha="center", va="center", color="black")

for i in range(cm_e.shape[0]):
    for j in range(cm_e.shape[1]):
        axes[1].text(j, i, cm_e[i, j], ha="center", va="center", color="black")

# Labels
for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()