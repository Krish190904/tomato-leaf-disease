import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "image_only_model.pth"  # or multimodal_model.pth
IMAGE_FOLDER = "dataset/field/test/Early_blight"
OUTPUT_FOLDER = "gradcam_outputs"
NUM_CLASSES = 5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- LOAD MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- GRADCAM CLASS ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = cam / np.max(cam)
        return cam

# ---------------- INIT GRADCAM ----------------
gradcam = GradCAM(model, model.layer4[-1])

# ---------------- PROCESS MULTIPLE IMAGES ----------------
for img_name in os.listdir(IMAGE_FOLDER):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(input_tensor)
        class_idx = torch.argmax(preds).item()

    cam = gradcam.generate(input_tensor, class_idx)

    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    save_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("✅ Grad-CAM generated for all images.")
