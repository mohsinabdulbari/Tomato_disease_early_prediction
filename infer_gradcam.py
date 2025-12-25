import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 160

pre_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = ckpt["classes"]

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, classes

def gradcam(model, x, idx):
    # hook last feature map
    model.features[18].register_forward_hook(
        lambda m, i, o: setattr(model, "feat", o)
    )
    model.zero_grad()
    out = model(x)
    out[0, idx].backward()

    fmap = model.feat.detach().cpu().numpy()[0]            # [C, H, W]
    grad = model.classifier[1].weight[idx].detach().cpu().numpy()  # [C]

    cam = np.tensordot(grad, fmap, axes=(0, 0))            # [H, W]
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam

def stage_and_severity(cam_resized):
    """
    cam_resized: [H,W] in [0,1]
    We treat very small hot areas as EARLY infection.
    """
    # more sensitive threshold (was 0.5 before)
    infected_pixels = np.sum(cam_resized > 0.3)
    total_pixels = cam_resized.size
    frac = infected_pixels / (total_pixels + 1e-8)  # 0..1
    severity = frac * 100.0                         # %

    if severity < 5:
        stage = "VERY EARLY / MINIMAL"
        risk = "LOW to MEDIUM"
    elif severity < 20:
        stage = "EARLY"
        risk = "HIGH"
    elif severity < 30:
        stage = "MODERATE"
        risk = "VERY HIGH"
    else:
        stage = "SEVERE"
        risk = "CRITICAL"

    return severity, stage, risk

def infer(img_path):
    model, classes = load_model()
    img = Image.open(img_path).convert("RGB")
    x = pre_tf(img).unsqueeze(0).to(DEVICE)

    # classification
    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out,1)[0]
    idx = prob.argmax().item()
    disease = classes[idx]
    conf = float(prob[idx])

    # gradcam & severity
    cam = gradcam(model, x, idx)
    cam_resized = cv2.resize(cam, (img.size[0], img.size[1]))
    severity, stage, risk = stage_and_severity(cam_resized)

    # ----- EARLY-DETECTION LOGIC FOR HEALTHY LEAVES -----
    is_healthy_label = disease.lower().startswith("healthy")
    early_warning = False
    early_note = "No"

    # if visually healthy but Grad-CAM sees small hotspot → possible early infection
    if is_healthy_label and severity > 3.0:
        early_warning = True
        early_note = "YES – Possible early infection region detected"
        # For reporting, override stage/risk to reflect suspicion
        stage = "POSSIBLE EARLY INFECTION"
        risk = "ELEVATED"

    # Visualization
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    save_path = "gradcam_output.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("\n🧠 Predicted Label:", disease)
    print("📊 Confidence:", round(conf, 4))
    print("🔥 Severity:", f"{severity:.2f}%")
    print("🔎 Stage:", stage)
    print("🚨 Future Risk:", risk)
    if is_healthy_label:
        print("🟢 Healthy Leaf Early Warning:", early_note)
    print("📌 Grad-CAM Saved:", save_path)

if __name__ == "__main__":
    infer("sample_leaf.jpg")