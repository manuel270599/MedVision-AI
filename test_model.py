import torch
import torch.nn as nn
import torchxrayvision as xrv
import cv2
import numpy as np
import os
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = xrv.models.DenseNet(in_channels=3, weights=None)
model.classifier = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 3)
)
model.load_state_dict(torch.load("models/densenet121_finetuned_pytorch.pth", map_location=DEVICE), strict=False)
model.eval()
model.to(DEVICE)

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar: {image_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=0)
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(DEVICE)

def predict(image_path, real_class):
    input_tensor = preprocess(image_path)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = ['Normal', 'Neumonia', 'Tuberculosis'][np.argmax(probs)]
    print(f"Clase real: {real_class:12} | Predicción: {pred_class:12} | Confianza: {np.max(probs):.2%} | Archivo: {os.path.basename(image_path)}")
    return pred_class == real_class

# Probar con imágenes de val
test_folders = {
    "Normal": "dataset/val/Normal/",
    "Neumonia": "dataset/val/Neumonia/",
    "Tuberculosis": "dataset/val/Tuberculosis/"
}

total = 0
correctas = 0

for clase, carpeta in test_folders.items():
    imagenes = glob.glob(os.path.join(carpeta, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(carpeta, "*.[jJ][pP][eE][gG]")) + \
                glob.glob(os.path.join(carpeta, "*.[pP][nN][gG]"))
    if not imagenes:
        print(f" No hay imágenes en {carpeta}")
        continue
    print(f"\n {clase} ({len(imagenes)} imágenes):")
    for img in imagenes[:5]:  #solo5 
        ok = predict(img, clase)
        total += 1
        correctas += ok

print(f"\n Precisión en muestra: {correctas}/{total} ({correctas/total:.2%})")