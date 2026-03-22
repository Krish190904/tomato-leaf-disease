import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os


# =====================================================
# GRAD-CAM CLASS
# =====================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=[2,3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam).detach().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam


# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =====================================================
# LOAD MODELS
# =====================================================
def load_resnet(model_path, num_classes=5):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()

def load_effnet(model_path, num_classes=5):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model.eval()


# =====================================================
# MAIN FUNCTION TO COMPARE
# =====================================================
def compare_gradcam(img_path, resnet_model_path, effnet_model_path):

    if not os.path.exists(img_path):
        print(f"❌ ERROR: Image not found → {img_path}")
        return

    if not os.path.exists(resnet_model_path):
        print(f"❌ ERROR: ResNet model not found → {resnet_model_path}")
        return

    if not os.path.exists(effnet_model_path):
        print(f"❌ ERROR: EfficientNet model not found → {effnet_model_path}")
        return

    print("✓ Loading image...")
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    print("✓ Loading ResNet18 and EfficientNet...")
    resnet = load_resnet(resnet_model_path)
    effnet = load_effnet(effnet_model_path)

    # target layers
    resnet_target = resnet.layer4[-1]
    effnet_target = effnet.features[-1]

    cam_resnet = GradCAM(resnet, resnet_target)
    cam_effnet = GradCAM(effnet, effnet_target)

    # Predictions
    pred_resnet = resnet(input_tensor).argmax(1).item()
    pred_effnet = effnet(input_tensor).argmax(1).item()

    # Heatmaps
    heat_resnet = cam_resnet.generate_heatmap(input_tensor, pred_resnet)
    heat_effnet = cam_effnet.generate_heatmap(input_tensor, pred_effnet)

    img_np = np.array(img)

    h1 = cv2.resize(heat_resnet, (img_np.shape[1], img_np.shape[0]))
    h2 = cv2.resize(heat_effnet, (img_np.shape[1], img_np.shape[0]))

    heat1 = cv2.applyColorMap(np.uint8(255 * h1), cv2.COLORMAP_JET)
    heat2 = cv2.applyColorMap(np.uint8(255 * h2), cv2.COLORMAP_JET)

    overlay1 = heat1 * 0.4 + img_np * 0.6
    overlay2 = heat2 * 0.4 + img_np * 0.6

    # Plot side-by-side
    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    plt.imshow(overlay1.astype("uint8"))
    plt.title(f"ResNet18 Grad-CAM\nPrediction = {pred_resnet}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay2.astype("uint8"))
    plt.title(f"EfficientNet Grad-CAM\nPrediction = {pred_effnet}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =====================================================
# USER INTERFACE (NO MANUAL EDITING)
# =====================================================
if __name__ == "__main__":
    print("\n=== GRAD-CAM SIDE-BY-SIDE COMPARISON ===")

    img_path = input("\nEnter path to field image: ").strip()
    resnet_model_path = input("Enter path to ResNet18 model (.pth): ").strip()
    effnet_model_path = input("Enter path to EfficientNet model (.pth): ").strip()

    compare_gradcam(img_path, resnet_model_path, effnet_model_path)