# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 18:10:37 2025

@author: msada
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== Define Model Class ==================
class DeepFakeModel(nn.Module):
    def __init__(self, base_model):
        super(DeepFakeModel, self).__init__()
        self.features = base_model
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        return x

# ================== LOAD MODEL ==================
def load_deepfake_model(model_path="./model.weights.pth"):
    base_model = models.densenet121(pretrained=False).features
    model = DeepFakeModel(base_model).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ================== INFERENCE ==================
def classify_deepfake(pil_image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = model(tensor).item()
        label = "Real" if prob > 0.5 else "Fake"
    return label, prob
# ================== EVALUATION ==================
def evaluate_folder(real_folder, fake_folder, model):
    y_true = []
    y_pred = []

    # Real images
    for filename in os.listdir(real_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(real_folder, filename)
            image = Image.open(path).convert("RGB")
            pred = classify_deepfake(image, model)
            y_true.append(1)  # Real = 1
            y_pred.append(pred)

    # Fake images
    for filename in os.listdir(fake_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(fake_folder, filename)
            image = Image.open(path).convert("RGB")
            pred = classify_deepfake(image, model)
            y_true.append(0)  # Fake = 0
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

# ================== RUN ==================
if __name__ == "__main__":
    model = load_deepfake_model("model.weights.pth")
    real_path = r"D:\\real_vs_fake\\real-vs-fake\\test\\real"
    fake_path = r"D:\\real_vs_fake\\real-vs-fake\\test\\fake"
    evaluate_folder(real_path, fake_path, model)


