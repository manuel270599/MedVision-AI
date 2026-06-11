# MEDVISION ASSIST

## Sistema Inteligente para el Análisis y Clasificación de Radiografías de Tórax mediante Procesamiento Digital de Imágenes y Redes Neuronales Convolucionales

Proyecto del curso de Computación Gráfica

## Estructura del Proyecto

```text
MedVision-AI/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── Recursos gráficos de la aplicación
│
├── Images/
│   └── simbolo-de-la-medicina.png
│
├── preprocessing/
│   └── enhancement.py
|   └── filters.py
|   └── histogram.py
│
├── segmentation/
│   └── classical.py
|   └── unet_segmentation.py
│
├── classification/
│   └── labels.py
│   └── predict.py
│
├── explainability/
│   └── gradcam.py
│
├── models/
│   └── classifier.pth
│   └── unet.pth
│
├── outputs/
│
├── utils/
│   └── image_utils.py
|   └── visualization.py
│
└── venv/
    └── Entorno virtual de desarrollo
```

---
