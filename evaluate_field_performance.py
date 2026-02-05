import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIELD_TEST_DIR = "dataset/field/test"
MODEL_PATH = "image_only_model.pth"
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
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

class_names = dataset.classes

# ---------------- LOAD MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- PREDICTIONS ----------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

# ---------------- CLASS-WISE ACCURACY ----------------
class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\nClass-wise Accuracy:")
for cls, acc in zip(class_names, class_accuracy):
    print(f"{cls}: {acc*100:.2f}%")

# ---------------- PLOT CONFUSION MATRIX ----------------
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Field Images")
plt.tight_layout()
plt.savefig("confusion_matrix_field.png")
plt.show()
