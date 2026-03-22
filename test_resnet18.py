


import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd


# ==============================
# CONFIG
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_DIR = "dataset/field/test"
ENV_CSV = "env_data.csv"

IMAGE_MODEL_PATH = "resnet18_image_only.pth"
MULTI_MODEL_PATH = "resnet18_multimodal_fixed.pth"   # <-- YOUR NEW TRAINED MODEL


# ==============================
# LOAD ENVIRONMENT CSV (FIXED)
# ==============================
env_df = pd.read_csv(ENV_CSV)
env_df["filename"] = env_df["path"].apply(lambda x: os.path.basename(str(x)))

# FIXED NORMALIZATION
for col in ["temperature", "humidity", "rainfall"]:
    mn, mx = env_df[col].min(), env_df[col].max()
    env_df[col] = (env_df[col] - mn) / (mx - mn)

env_dict = {
    row["filename"]: torch.tensor(
        [row["temperature"], row["humidity"], row["rainfall"]], dtype=torch.float32
    )
    for _, row in env_df.iterrows()
}

DEFAULT_ENV = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)


# ==============================
# CUSTOM DATASET WITH FILE PATHS
# ==============================
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


# ==============================
# TEST TRANSFORM
# ==============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_ds = ImageFolderWithPaths(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
class_names = test_ds.classes


# ==============================
# RESNET18 IMAGE-ONLY MODEL
# ==============================
def load_image_only_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==============================
# RESNET18 MULTIMODAL MODEL
# ==============================
class ResNet18_Multimodal(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()      # 512 features

        self.env_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, env):
        img_feat = self.cnn(img)
        env_feat = self.env_net(env)
        fused = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(fused)


def load_multimodal_resnet18():
    model = ResNet18_Multimodal(len(class_names))
    model.load_state_dict(torch.load(MULTI_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==============================
# TEST FUNCTIONS
# ==============================
def test_image_only(model):
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label, _ in test_loader:
            img = img.to(DEVICE)
            output = model(img)
            pred = output.argmax(1).item()

            y_true.append(label.item())
            y_pred.append(pred)

    return y_true, y_pred


def test_multimodal(model):
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label, path in test_loader:
            img = img.to(DEVICE)

            fname = os.path.basename(path[0])
            env_vec = env_dict.get(fname, DEFAULT_ENV).unsqueeze(0).to(DEVICE)

            output = model(img, env_vec)
            pred = output.argmax(1).item()

            y_true.append(label.item())
            y_pred.append(pred)

    return y_true, y_pred


# ==============================
# CONFUSION MATRIX
# ==============================
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Greens", xticks_rotation=45)
    plt.title(title)
    plt.show()

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"{title} Accuracy: {acc:.4f}")


# ==============================
# RUN BOTH TESTS
# ==============================
if __name__ == "__main__":

    print("\n==============================")
    print(" TESTING: ResNet18 Image-Only ")
    print("==============================")

    img_model = load_image_only_resnet18()
    y_true_img, y_pred_img = test_image_only(img_model)
    plot_confusion(y_true_img, y_pred_img, "ResNet18 Image-Only (Field Accuracy)")

    print("\n==============================")
    print(" TESTING: ResNet18 Multimodal ")
    print("==============================")

    multi_model = load_multimodal_resnet18()
    y_true_mul, y_pred_mul = test_multimodal(multi_model)
    plot_confusion(y_true_mul, y_pred_mul, "ResNet18 Multimodal (Field Accuracy)")