import torch
import torch.nn as nn
import torchxrayvision as xrv
import numpy as np
import cv2
import streamlit as st

# Configurar dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """
    Carga el modelo fine-tuned desde el archivo local.
    - Crea el modelo con in_channels=3 y sin pesos preentrenados (weights=None).
    - Carga directamente los pesos del archivo fine-tuned.
    - Si falla, usa el modelo base con pesos preentrenados (mapeo de 18 clases).
    """
    try:
        #  Crear modelo con 3 canales, SIN pesos preentrenados 
        model = xrv.models.DenseNet(in_channels=3, weights=None)
        

        model.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)  # 3 clases: Normal, Neumonía, Tuberculosis
        )
        

        model.load_state_dict(
            torch.load("models/densenet121_finetuned_pytorch.pth", map_location=DEVICE),
            strict=False  
        )
        st.success("- Modelo fine-tuned cargado correctamente.")
        
    except FileNotFoundError:
        st.warning("- No se encontró modelo fine-tuned. Usando modelo base (mapeo de 18 clases).")
        model = xrv.models.DenseNet(in_channels=3, weights="densenet121-res224-all")
    except Exception as e:
        st.warning(f"- Error al cargar modelo fine-tuned: {e}. Usando modelo base.")
        # pesos preentrenados
        model = xrv.models.DenseNet(in_channels=3, weights="densenet121-res224-all")
    
    model.eval()
    model.to(DEVICE)
    return model

def preprocess_for_ai(image_gray):
    resized = cv2.resize(image_gray, (224, 224))
    img = resized.astype(np.float32) / 255.0
    img_rgb = np.stack([img, img, img], axis=2)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor.to(DEVICE)

def predict(model, input_tensor):
    """
    Ejecuta la inferencia y retorna las probabilidades.
    Si el modelo es fine-tuned (3 salidas), devuelve softmax.
    Si es el modelo base (18 salidas), devuelve sigmoid.
    """
    with torch.no_grad():
        logits = model(input_tensor)
        if logits.shape[1] == 3:
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        return probs

def map_to_3_classes(probs_18):
    """
    Mapea las 18 salidas del modelo base a 3 clases.
    """
    if len(probs_18) == 3:
        return probs_18
    pneumonia_prob = probs_18[8]  # neumonia
    tb_prob = max(probs_18[1], probs_18[2])  
    max_pathology = max(pneumonia_prob, tb_prob)
    normal_prob = max(0.0, 1.0 - max_pathology)
    total = normal_prob + pneumonia_prob + tb_prob
    if total > 0:
        normal_prob /= total
        pneumonia_prob /= total
        tb_prob /= total
    else:
        normal_prob = 1.0
        pneumonia_prob = 0.0
        tb_prob = 0.0
    return [normal_prob, pneumonia_prob, tb_prob]   