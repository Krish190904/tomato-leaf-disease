from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

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

# ---------------- MODEL ----------------
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


model = MultiModalNet().to(DEVICE)
model.load_state_dict(torch.load("multimodal_model.pth", map_location=DEVICE))
model.eval()

# ---------------- PREPROCESS ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- FASTAPI ----------------
app = FastAPI(title="Tomato Leaf Disease Prediction API")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    temperature: float = 25,
    humidity: float = 70,
    rainfall: float = 5
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    env = torch.tensor([[temperature, humidity, rainfall]],
                       dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(image, env)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "disease": CLASS_NAMES[pred.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
