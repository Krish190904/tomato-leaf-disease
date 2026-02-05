import os
import torch
import pandas as pd
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIELD_TEST_DIR = "dataset/field/test"
MODEL_PATH = "multimodal_model.pth"
ENV_CSV = "env_data.csv"
NUM_CLASSES = 5

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- LOAD DATA ----------------
dataset = datasets.ImageFolder(FIELD_TEST_DIR, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
class_names = dataset.classes

# ---------------- LOAD ENV DATA ----------------
env_df = pd.read_csv(ENV_CSV)

# Extract filename from full path
env_df["filename"] = env_df["path"].apply(lambda x: os.path.basename(str(x)))

env_dict = {
    row["filename"]: torch.tensor(
        [row["temperature"], row["humidity"], row["rainfall"]],
        dtype=torch.float32
    )
    for _, row in env_df.iterrows()
}

# ---------------- MULTIMODAL MODEL ----------------
class MultimodalNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Image branch
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = torch.nn.Identity()

        # Environmental branch (MATCHES TRAINING)
        self.env_net = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU()
        )

        # Classifier (MATCHES TRAINING)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 + 32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, img, env):
        img_feat = self.cnn(img)
        env_feat = self.env_net(env)
        combined = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(combined)

# ---------------- LOAD MODEL ----------------
model = MultimodalNet(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- EVALUATION ----------------
y_true, y_pred = [], []
sample_idx = 0

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)

        batch_paths = loader.dataset.samples[sample_idx:sample_idx + len(imgs)]
        sample_idx += len(imgs)

        envs = []
        for path, _ in batch_paths:
            fname = os.path.basename(path)
            if fname in env_dict:
                envs.append(env_dict[fname])
            else:
                # Safe fallback values
                envs.append(torch.tensor([25.0, 60.0, 5.0]))

        envs = torch.stack(envs).to(DEVICE)

        outputs = model(imgs, envs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

# ---------------- CLASS-WISE ACCURACY ----------------
class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nClass-wise Accuracy (Multimodal Model):")
for cls, acc in zip(class_names, class_acc):
    print(f"{cls}: {acc*100:.2f}%")

# ---------------- PLOT ----------------
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Greens")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Multimodal (Field Images)")
plt.tight_layout()
plt.savefig("confusion_matrix_multimodal.png")
plt.show()
