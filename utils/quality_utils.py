import cv2
import numpy as np

def compute_metrics(image):
    """
    Calcula métricas de calidad de la imagen.
    Retorna: contraste (std), nitidez (varianza del Laplaciano), entropía.
    """
    contrast = np.std(image)
    gradient = cv2.Laplacian(image, cv2.CV_64F).var()
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return contrast, gradient, entropy

def get_quality_level(contrast, gradient):
    """
    Determina el nivel de calidad basado en umbrales empíricos.
    Retorna: 'reject', 'warning', o 'optimal'.
    """
    if contrast < 15 or gradient < 5.0:
        return 'reject'
    elif contrast < 30 or gradient < 10.0:
        return 'warning'
    else:
        return 'optimal'

def apply_auto_enhancement(image_gray):
    """
    Mejora automática de la imagen:
    1. CLAHE suave (clipLimit=1.5) para mejorar contraste local.
    2. Filtro Bilateral para reducir ruido preservando bordes nítidos.
    """
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_gray)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised