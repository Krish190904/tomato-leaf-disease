# 🍅 FarmGuard AI — Tomato Leaf Disease Detection

An AI-powered web application that detects **5 common tomato leaf diseases** using a multimodal deep learning model. It combines leaf image analysis with environmental weather data (temperature, humidity, rainfall) to deliver more accurate diagnoses, along with severity ratings and actionable treatment recommendations for farmers.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **🔬 Multimodal AI** — Fuses leaf images + weather data for higher accuracy than image-only models
- **📊 5 Disease Classes** — Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Yellow Leaf Curl Virus
- **🎯 Severity Assessment** — Classifies severity as Low / Medium / High based on confidence thresholds
- **💊 Treatment Plans** — Returns detailed, crop-specific treatment recommendations per disease
- **🌿 Leaf Validation** — Uses green-pixel ratio (HSV) to reject non-leaf uploads before prediction
- **📱 Mobile-First UI** — Step-by-step wizard (Upload → Weather → Results) optimized for farmers in the field
- **⚡ Real-Time Inference** — Runs on CPU with no GPU required

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (HTML/CSS/JS)          │
│        3-Step Wizard: Upload → Weather → Result  │
└──────────────────────┬──────────────────────────┘
                       │ POST /predict
                       ▼
┌─────────────────────────────────────────────────┐
│               FastAPI Backend (main.py)          │
│  ┌───────────┐  ┌────────────┐  ┌────────────┐  │
│  │ Leaf      │  │ Image      │  │ Env Data   │  │
│  │ Validator │  │ Transform  │  │ Normalizer │  │
│  │ (OpenCV)  │  │ (torchvision)│ │ (min-max)  │  │
│  └─────┬─────┘  └──────┬─────┘  └──────┬─────┘  │
│        │               │               │         │
│        ▼               ▼               ▼         │
│  ┌─────────────────────────────────────────────┐ │
│  │     EfficientNet-B0 Multimodal Model        │ │
│  │  ┌──────────────┐   ┌────────────────────┐  │ │
│  │  │ CNN (1280-d)  │   │ Env MLP (3→16→32) │  │ │
│  │  └──────┬───────┘   └────────┬───────────┘  │ │
│  │         └───── Concat (1312) ─┘              │ │
│  │                    │                          │ │
│  │           Classifier (1312→128→5)             │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
faarzi/
├── main.py                          # 🚀 FastAPI server (production entry point)
├── requirements.txt                 # Python dependencies
├── efficientnet_multimodal_best.pth # Trained model weights (~17MB)
├── env_data.csv                     # Environmental data used in training
├── templates/
│   └── index.html                   # Frontend UI (embedded CSS + JS)
├── static/
│   ├── style.css                    # Additional styles
│   └── uploads/                     # Uploaded images (auto-created)
├── dataset/
│   ├── lab/                         # Lab-sourced training images
│   └── field/                       # Field-captured test images
│
│── # ─── Training & Evaluation Scripts (one-time use) ───
├── efficientnet_image_only.py       # Training: EfficientNet image-only
├── efficientnet_multimodal.py       # Training: EfficientNet multimodal
├── train_resnet18_image_only.py     # Training: ResNet18 image-only
├── train_resnet18_multimodal.py     # Training: ResNet18 multimodal
├── test_resnet18.py                 # Evaluation: ResNet18 models
├── evaluate_efficientnet_field.py   # Evaluation: EfficientNet models
├── confu.py                         # Confusion matrix comparison
└── gradcam_compare_resnet_vs_efficientnet.py  # Grad-CAM visualization
```

> **Note:** The training/evaluation scripts were used during development. Only `main.py` is needed to run the app.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/farmguard-ai.git
cd farmguard-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python main.py
```

The app will start at **http://localhost:8000**

---

## 🔌 API Reference

### `POST /predict`

Upload a tomato leaf image with weather data to get a disease prediction.

**Request** (`multipart/form-data`):

| Field         | Type   | Default | Description                     |
|---------------|--------|---------|---------------------------------|
| `file`        | File   | —       | Leaf image (JPG/PNG)            |
| `temperature` | float  | 25.0    | Temperature in °C               |
| `humidity`    | float  | 70.0    | Humidity percentage              |
| `rainfall`    | string | "low"   | One of: `low`, `moderate`, `high` |

**Response** (JSON):

```json
{
  "prediction": "Early Blight",
  "confidence": 0.9234,
  "confidence_pct": 92.3,
  "severity": "High",
  "emoji": "🟤",
  "description": "A common fungal disease caused by Alternaria solani...",
  "cause": "Warm, humid weather (24-29°C)...",
  "treatment": [
    "Remove and destroy infected lower leaves immediately",
    "Apply fungicide: Mancozeb (2.5g/L) or Chlorothalonil spray",
    "..."
  ],
  "model_used": "EfficientNet Multimodal"
}
```

---

## 🧠 Model Details

| Property          | Value                          |
|-------------------|--------------------------------|
| **Backbone**      | EfficientNet-B0 (ImageNet pretrained) |
| **Environmental Input** | Temperature, Humidity, Rainfall (3-dim MLP) |
| **Fusion**        | Feature concatenation (1280 + 32 = 1312) |
| **Classifier**    | 2-layer MLP (1312 → 128 → 5)  |
| **Training**      | 3-stage progressive unfreezing |
| **Image Size**    | 224 × 224 px                   |
| **Normalization** | ImageNet (mean/std)            |

### Diseases Detected

| # | Disease               | Pathogen Type |
|---|-----------------------|---------------|
| 1 | Early Blight          | Fungal        |
| 2 | Late Blight           | Oomycete      |
| 3 | Leaf Mold             | Fungal        |
| 4 | Septoria Leaf Spot    | Fungal        |
| 5 | Yellow Leaf Curl Virus| Viral         |

---

## 🛠️ Tech Stack

| Layer     | Technology                      |
|-----------|---------------------------------|
| Backend   | Python, FastAPI, Uvicorn        |
| AI/ML     | PyTorch, TorchVision, EfficientNet-B0 |
| Vision    | OpenCV (leaf validation), Pillow |
| Frontend  | HTML5, CSS3, Vanilla JavaScript |
| Design    | Mobile-first, glassmorphism, Inter font |

---

## 📜 License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ❤️ for farmers · Powered by <strong>EfficientNet AI</strong> 🌱
</p>
