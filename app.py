from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import cv2
import numpy as np
import io
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 160

pre_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


model = None
classes = None

@app.on_event("startup")
def load_model():
    global model, classes
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()


def gradcam(x, idx):
    model.features[18].register_forward_hook(
        lambda m, i, o: setattr(model, "feat", o)
    )
    model.zero_grad()
    pred = model(x)
    pred[0, idx].backward()

    fmap = model.feat.detach().cpu().numpy()[0]
    grad = model.classifier[1].weight[idx].detach().cpu().numpy()
    cam = np.tensordot(grad, fmap, axes=(0, 0))
    cam = np.maximum(cam, 0)
    return cam / cam.max()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_np = np.array(img)

    # ---------- LEAF VALIDATION ----------
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_percentage = (mask_green > 0).mean() * 100

    if green_percentage < 15:
        return {
            "error": True,
            "message": "❌ Not a tomato leaf! Upload a clear tomato leaf image."
        }

    # Prediction
    x = pre_tf(img).unsqueeze(0).to(DEVICE)
    out = model(x)
    prob = F.softmax(out, 1)[0]
    confidence, idx = torch.max(prob, 0)
    confidence = float(confidence)

    # -------- Reject Low Confidence ---------
    if confidence < 0.60:
        return {
            "error": True,
            "message": "⚠️ Image unclear — retake a clear picture of a leaf."
        }

    disease = classes[idx]

    cam = gradcam(x, idx)
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    os.makedirs("static", exist_ok=True)
    save_path = "static/gradcam_output.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    infected_pixels = np.sum(cam > 0.5)
    total_pixels = cam.size
    severity = (infected_pixels / total_pixels) * 100

    # ----- Healthy Fix -----
    if disease.lower() == "healthy" and confidence > 0.90:
        severity = 5.0
        stage = "SAFE"
        risk = "LOW"
    else:
        if severity < 20:
            stage = "EARLY"
            risk = "MEDIUM"
        elif severity < 50:
            stage = "MODERATE"
            risk = "HIGH"
        else:
            stage = "SEVERE"
            risk = "CRITICAL"

    return {
        "error": False,
        "disease": disease,
        "confidence": confidence,
        "severity": round(severity, 2),
        "stage": stage,
        "future_risk": risk,
        "gradcam": "/static/gradcam_output.jpg"
    }