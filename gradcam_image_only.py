import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

CLASS_NAMES = [
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Yellow_Leaf_Curl_Virus"
]

# ---------------- LOAD IMAGE-ONLY MODEL ----------------
# IMPORTANT: This matches how the model was trained
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 5)

model.load_state_dict(
    torch.load("image_only_model.pth", map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

# Target layer for Grad-CAM
target_layer = model.layer4[-1]

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- GRAD-CAM CLASS ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(img_tensor)
        output[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam / cam.max()
        return cam

gradcam = GradCAM(model, target_layer)

# ---------------- LOAD IMAGE ----------------
IMAGE_PATH = "C:/Users/uroha/OneDrive/Desktop/faarzi/dataset/field/test/Early_blight/1234080-Early-Blight.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ---------------- PREDICT ----------------
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

# ---------------- GENERATE CAM ----------------
cam = gradcam.generate(input_tensor, pred_class)

# ---------------- VISUALIZE ----------------
img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grad-CAM (Image-only)")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title(f"Overlay ({CLASS_NAMES[pred_class]})")
plt.imshow(overlay)
plt.axis("off")

plt.show()