import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import uuid

app = Flask(__name__)

DEVICE = "cpu"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = [
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Yellow Leaf Curl Virus"
]


# ---------------------------------------------------
# DISEASE KNOWLEDGE BASE
# ---------------------------------------------------

DISEASE_INFO = {
    "Early Blight": {
        "emoji": "🟤",
        "description": "A common fungal disease caused by Alternaria solani. It appears as dark brown spots with concentric rings (target-like pattern) on older, lower leaves first.",
        "cause": "Warm, humid weather (24-29°C) with alternating wet and dry periods. Spreads through wind, rain splash, and contaminated soil.",
        "treatment": [
            "Remove and destroy infected lower leaves immediately",
            "Apply fungicide: Mancozeb (2.5g/L) or Chlorothalonil spray",
            "Use copper-based fungicide (Copper oxychloride 3g/L) as preventive",
            "Ensure proper spacing between plants for air circulation",
            "Avoid overhead watering — use drip irrigation instead",
            "Rotate crops — do not plant tomatoes in same spot for 2-3 years"
        ],
        "severity_thresholds": {"low": 0.5, "medium": 0.75}
    },
    "Late Blight": {
        "emoji": "⚫",
        "description": "A devastating disease caused by Phytophthora infestans. It causes large, dark, water-soaked patches on leaves and stems. Can destroy an entire crop within days.",
        "cause": "Cool, wet weather (15-22°C) with high humidity (above 80%). Spreads very rapidly through wind-carried spores.",
        "treatment": [
            "ACT IMMEDIATELY — this disease spreads very fast!",
            "Remove and burn all infected plants (do NOT compost)",
            "Apply systemic fungicide: Metalaxyl + Mancozeb (Ridomil Gold 2.5g/L)",
            "Spray preventive fungicide on healthy nearby plants",
            "Improve drainage — avoid waterlogged conditions",
            "Use resistant tomato varieties for next season",
            "Alert neighboring farms — spores travel by wind"
        ],
        "severity_thresholds": {"low": 0.4, "medium": 0.65}
    },
    "Leaf Mold": {
        "emoji": "🟢",
        "description": "Caused by the fungus Passalora fulva (formerly Cladosporium fulvum). Shows pale green to yellowish spots on upper leaf surface with olive-green to brown velvety mold underneath.",
        "cause": "High humidity (above 85%) and moderate temperatures (20-25°C). Common in greenhouses and poorly ventilated areas.",
        "treatment": [
            "Improve ventilation — open greenhouse vents or space plants wider",
            "Reduce humidity below 85% if possible",
            "Remove heavily infected leaves carefully",
            "Apply fungicide: Chlorothalonil or copper-based spray",
            "Avoid wetting leaves during watering",
            "Prune lower branches to improve airflow",
            "Use resistant varieties (with Cf gene resistance)"
        ],
        "severity_thresholds": {"low": 0.5, "medium": 0.7}
    },
    "Septoria Leaf Spot": {
        "emoji": "⚪",
        "description": "Caused by the fungus Septoria lycopersici. Appears as many small, circular spots (1-3mm) with dark brown borders and gray-white centers, often with tiny black dots (pycnidia).",
        "cause": "Warm, wet weather (20-25°C) with frequent rain or overhead irrigation. Splashing water spreads spores from soil to lower leaves.",
        "treatment": [
            "Remove infected lower leaves and destroy them",
            "Apply fungicide: Mancozeb or Chlorothalonil (start early)",
            "Mulch around plants to prevent soil splash",
            "Use drip irrigation instead of overhead sprinklers",
            "Stake or cage plants to keep leaves off the ground",
            "Practice crop rotation (3-year cycle)",
            "Clean garden tools between plants to prevent spread"
        ],
        "severity_thresholds": {"low": 0.5, "medium": 0.72}
    },
    "Yellow Leaf Curl Virus": {
        "emoji": "🟡",
        "description": "A viral disease (TYLCV) transmitted by whiteflies (Bemisia tabaci). Causes upward curling of leaf edges, yellowing, stunted growth, and reduced fruit production.",
        "cause": "Spread by whitefly insects. Hot, dry weather increases whitefly populations. The virus cannot be cured once a plant is infected.",
        "treatment": [
            "NO CURE exists — focus on prevention and management",
            "Remove and destroy infected plants immediately",
            "Control whiteflies: use yellow sticky traps around plants",
            "Apply neem oil spray (5ml/L) to repel whiteflies",
            "Use insecticide: Imidacloprid for severe whitefly infestations",
            "Cover seedlings with insect-proof mesh (40 mesh net)",
            "Plant resistant/tolerant varieties (Ty gene varieties)",
            "Grow reflective mulch to repel whiteflies"
        ],
        "severity_thresholds": {"low": 0.4, "medium": 0.6}
    }
}


# ---------------------------------------------------
# NORMALIZATION VALUES (MUST MATCH TRAINING)
# ---------------------------------------------------

TEMP_MIN = 10
TEMP_MAX = 40

HUM_MIN = 20
HUM_MAX = 100

RAIN_MIN = 0
RAIN_MAX = 60


def normalize_env(temp, hum, rain):

    temp = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    hum = (hum - HUM_MIN) / (HUM_MAX - HUM_MIN)
    rain = (rain - RAIN_MIN) / (RAIN_MAX - RAIN_MIN)

    return temp, hum, rain


# ---------------------------------------------------
# IMAGE TRANSFORM (FIXED: added ImageNet normalization)
# ---------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------------------------------------------
# LEAF VALIDATION
# ---------------------------------------------------

def is_leaf_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    lower_green = np.array([25,40,40])
    upper_green = np.array([90,255,255])

    mask = cv2.inRange(hsv,lower_green,upper_green)

    green_ratio = np.sum(mask>0)/(224*224)

    return green_ratio > 0.10


# ---------------------------------------------------
# IMAGE-ONLY MODELS
# ---------------------------------------------------

def load_resnet_image():

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,5)

    model.load_state_dict(
        torch.load("resnet18_image_only.pth",map_location=DEVICE)
    )

    model.eval()
    return model


def load_efficient_image():

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features,5)

    model.load_state_dict(
        torch.load("efficientnet_image_only_improved.pth",map_location=DEVICE)
    )

    model.eval()
    return model


# ---------------------------------------------------
# MULTIMODAL MODELS
# ---------------------------------------------------

class MultimodalResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        self.env_net = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512+32,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )

    def forward(self,img,env):

        img_feat = self.cnn(img)
        env_feat = self.env_net(env)

        fused = torch.cat([img_feat,env_feat],dim=1)

        return self.classifier(fused)


class MultimodalEfficientNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = models.efficientnet_b0(weights=None)
        self.cnn.classifier[1] = nn.Identity()

        self.env_net = nn.Sequential(
            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1280+32,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )

    def forward(self,img,env):

        img_feat = self.cnn(img)
        env_feat = self.env_net(env)

        fused = torch.cat([img_feat,env_feat],dim=1)

        return self.classifier(fused)


def load_resnet_multimodal():

    model = MultimodalResNet()

    model.load_state_dict(
        torch.load("resnet18_multimodal_fixed.pth",map_location=DEVICE)
    )

    model.eval()
    return model


def load_efficient_multimodal():

    model = MultimodalEfficientNet()

    model.load_state_dict(
        torch.load("efficientnet_multimodal_best.pth",map_location=DEVICE)
    )

    model.eval()
    return model


# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

resnet_image = load_resnet_image()
efficient_image = load_efficient_image()

resnet_multi = load_resnet_multimodal()
efficient_multi = load_efficient_multimodal()


# ---------------------------------------------------
# PREDICTION FUNCTIONS (ENHANCED — return structured data)
# ---------------------------------------------------

def get_severity(confidence, disease_name):
    """Map confidence to severity level."""
    thresholds = DISEASE_INFO[disease_name]["severity_thresholds"]
    if confidence < thresholds["low"]:
        return "Low"
    elif confidence < thresholds["medium"]:
        return "Medium"
    else:
        return "High"


def predict_image_detailed(model, img_tensor):

    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    disease = classes[pred.item()]
    conf = round(confidence.item(), 4)
    info = DISEASE_INFO[disease]

    return {
        "disease": disease,
        "confidence": conf,
        "confidence_pct": round(conf * 100, 1),
        "severity": get_severity(conf, disease),
        "emoji": info["emoji"],
        "description": info["description"],
        "cause": info["cause"],
        "treatment": info["treatment"]
    }


def predict_multimodal_detailed(model, img_tensor, env_tensor):

    with torch.no_grad():
        out = model(img_tensor, env_tensor)
        probs = torch.softmax(out, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    disease = classes[pred.item()]
    conf = round(confidence.item(), 4)
    info = DISEASE_INFO[disease]

    return {
        "disease": disease,
        "confidence": conf,
        "confidence_pct": round(conf * 100, 1),
        "severity": get_severity(conf, disease),
        "emoji": info["emoji"],
        "description": info["description"],
        "cause": info["cause"],
        "treatment": info["treatment"]
    }


# ---------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """JSON API endpoint for predictions."""

    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        # Save with unique name to avoid collisions
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(path)

        # Leaf validation
        if not is_leaf_image(path):
            return jsonify({
                "error": "This doesn't look like a tomato leaf. Please take a clear photo of a single tomato leaf against a plain background."
            }), 400

        # Read form fields
        temp = float(request.form.get("temperature", 25))
        hum = float(request.form.get("humidity", 70))
        rainfall = request.form.get("rainfall", "low")

        # Rainfall conversion
        if rainfall == "low":
            rain = 5
        elif rainfall == "moderate":
            rain = 25
        else:
            rain = 50

        # Prepare image tensor
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # Normalize environmental inputs
        temp_n, hum_n, rain_n = normalize_env(temp, hum, rain)
        env_tensor = torch.tensor([[temp_n, hum_n, rain_n]], dtype=torch.float32)

        # Use EfficientNet Multimodal (best model) by default
        result = predict_multimodal_detailed(efficient_multi, img_tensor, env_tensor)
        result["image_path"] = path
        result["model_used"] = "EfficientNet Multimodal"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)