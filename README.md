# MedVision Assist

> Sistema Inteligente para el Análisis y Clasificación de Radiografías de Tórax mediante Procesamiento Digital de Imágenes y Redes Neuronales Convolucionales.

## Descripción

**MedVision Assist** es un proyecto desarrollado como parte del curso **Computación gráfica** de la **Facultad de Ciencias** de la **Universidad Nacional de Ingenieria**, cuyo propósito es integrar técnicas de visión por computadora e inteligencia artificial para apoyar el análisis preliminar de radiografías de tórax.

El proyecto asume métodos clásicos de procesamiento digital de imágenes con modelos de aprendizaje profundo para mejorar la calidad visual de las radiografías y realizar una clasificación automática de enfermedades respiratorias. A lo largo de su desarrollo se abordaron aspectos relacionados con el tratamiento de imágenes médicas, la extracción de información cuantitativa, el diseño de interfaces de usuario y la aplicación de redes neuronales convolucionales mediante aprendizaje por transferencia.

## Sobre esta documentación

Esta documentación no solo presenta el resultado final del proyecto, sino también el proceso seguido durante su desarrollo. Cada sección explica las decisiones de diseño, las tecnologías empleadas y la justificación de las soluciones implementadas, permitiendo comprender la evolución del sistema desde su concepción hasta su estado actual.

El objetivo es ofrecer una visión clara y estructurada del proyecto, facilitando tanto su comprensión como su futura extensión o mantenimiento.
---

# Índice

1. Problema
2. Propuesta
3. Objetivos
4. Estructura del proyecto
5. Flujo de trabajo
6. Tecnologías Utilizadas
7. Arquitectura 
8. Construcción del Sistema
   - Interfaz
   - Procesamiento Digital de Imágenes
   - Histogramas
   - Métricas
   - Interpretación Automática
   - Explicación del Código Fuente (Archivos y Líneas Clave)**
       - `utils/quality_utils.py` – Filtro de Admisión
       - `utils/model_utils.py` – Carga del Modelo y Preprocesamiento para IA
       - `utils/gradcam_utils.py` – Explicabilidad Visual
       - `app.py` – Orquestación y Lógica de Decisión
       - `train.py` – Estrategia de Entrenamiento y Fine-Tuning
9. Modelo de Inteligencia Artificial
10. Capturas y evidencia
11. Ejecución del proyecto
12. Trabajo Futuro
13. Créditos



## 1. Problema

Las enfermedades respiratorias, como la neumonía y la tuberculosis, representan un importante problema de salud pública debido a su alta incidencia y mortalidad a nivel mundial. En este contexto, la radiografía de tórax constituye uno de los principales métodos de apoyo para la detección y evaluación de estas patologías; sin embargo, su efectividad depende en gran medida de la calidad de la imagen obtenida y de la correcta interpretación por parte del personal médico.

Uno de los principales inconvenientes es que las radiografías pueden presentar deficiencias de calidad ocasionadas por factores técnicos durante su adquisición, tales como bajo contraste, niveles inadecuados de brillo, presencia de ruido o artefactos e incluso diferencias entre equipos de radiología. Estas condiciones pueden dificultar la visualización de estructuras anatómicas importantes y reducir la confiabilidad del análisis clínico.

A estas limitaciones se suman los desafíos propios de la interpretación humana. La elevada carga de trabajo de los especialistas, la subjetividad inherente al análisis visual y la escasez de radiólogos en diversas regiones pueden incrementar la probabilidad de errores, generar diferencias en los diagnósticos y retrasar la atención de los pacientes.

Frente a esta problemática, surge la necesidad de desarrollar herramientas tecnológicas que permitan mejorar la calidad de las radiografías y proporcionar un apoyo objetivo durante su análisis. En este sentido, el proyecto **MedVision Assist** busca contribuir a esta necesidad mediante la integración de técnicas de Procesamiento Digital de Imágenes e Inteligencia Artificial, permitiendo optimizar la calidad visual de las radiografías, obtener métricas objetivas sobre la imagen y realizar una clasificación automática de enfermedades respiratorias como **Normal**, **Neumonía** y **Tuberculosis**. De esta manera, el sistema pretende servir como una herramienta de apoyo para los profesionales de la salud, facilitando el análisis preliminar de las imágenes y contribuyendo a una toma de decisiones más rápida, consistente y fundamentada.


## 2. Propuesta

La solución propuesta es MedVision Assist: un sistema inteligente híbrido que combina un robusto pipeline de procesamiento digital de imágenes (preprocesamiento) con un clasificador basado en aprendizaje profundo
(Deep Learning) para la identificación preliminar de neumonía, tuberculosis o normalidad en radiografías de tórax.



## 3. Objetivos

### Objetivo general
El proyecto tiene como objetivo general desarrollar un sistema inteligente que integre técnicas de procesamiento digital de imágenes y redes neuronales convolucionales para analizar radiografías de tórax y apoyar la clasificación preliminar de enfermedades pulmonares.

### Objetivos específicos
- La implementación de un pipeline robusto de mejora de imagen, el cual incluye ecualización CLAHE y filtros espaciales (Gaussiano y de Mediana), permitiendo corregir problemas de contraste y ruido en las radiografías.
- El desarrollo de herramientas de análisis visual (histogramas interactivos) y extracción de métricas de calidad (contraste, nitidez y entropía) que diagnostican automáticamente la fiabilidad de la imagen de entrada.
- La integración del pipeline de procesamiento y las visualizaciones en una interfaz interactiva accesible mediante Streamlit, facilitando su uso por parte del personal médico sin conocimientos técnicos.
- El entrenamiento de una red DenseNet121 mediante Transfer Learning y Fine-Tuning sobre un dataset combinado (ChestX-ray2017 y Montgomery County), logrando la clasificación automática en tres categorías: **Normal**, **Neumonía** y **Tuberculosis**.
- La implementación de mapas de calor explicativos (Grad-CAM) que superponen las regiones de activación de la red sobre la radiografía original, brindando transparencia y soporte visual al diagnóstico emitido por el modelo.
- La evaluación del rendimiento clínico del modelo, obteniendo métricas de Accuracy, Precisión, Recall y F1-Score superiores al **85%** en la detección combinada, cumpliendo con los estándares de la literatura especializada.


## 4. Estructura del Proyecto

```text
MedVision-AI/
├── app.py                     # Interfaz principal
├── train.py                   # Entrenamiento del modelo
├── organize_dataset.py        # Organización del dataset
├── test_model.py              # Testing parcial
├── requirements.txt           # Dependencias
├── dataset/                   # Imágenes (train/val)
│   ├── train/
│   │   ├── Normal/
│   │   ├── Neumonia/
│   │   └── Tuberculosis/
│   └── val/
│       ├── Normal/
│       ├── Neumonia/
│       └── Tuberculosis/
├── models/                    # Modelos guardados
│   ├── densenet121_finetuned_pytorch.pth
│   └── backup_model.pth
├── utils/                     # Módulos auxiliares
│   ├── model_utils.py
│   ├── quality_utils.py
│   └── gradcam_utils.py
├── Images/                    # Recursos visuales
│   └── simbolo_de_la_medicina.png
├── README.md
├── .gitignore
└── venv/                 
```
## 5. Flujo de trabajo

1. **El usuario sube una radiografía** (JPG, JPEG, PNG).
2. **El sistema evalúa la calidad original** (contraste, nitidez, entropía) y asigna un estado (`reject`, `warning`, `optimal`).
3. **Se aplica mejora automática** (CLAHE + Filtro Bilateral) y se re-evalúa la calidad.
4. **Si la calidad es óptima o aceptable**, la imagen se envía al modelo IA.
5. **El modelo DenseNet121** clasifica la imagen en **Normal**, **Neumonía** o **Tuberculosis**.
6. **Se muestra el diagnóstico** con la confianza ajustada (penalización del 20% si la calidad es `warning`).
7. **Se genera un mapa de calor Grad-CAM** para visualizar las regiones que influyeron en la decisión.

## 6. Tecnologías Utilizadas

| Tecnología | Uso |
| :--- | :--- |
| **Python** | Lenguaje principal. |
| **OpenCV** | Procesamiento de imágenes (CLAHE, filtros, lectura). |
| **PyTorch** | Framework de Deep Learning. |
| **TorchXRayVision** | Modelo DenseNet121 preentrenado en radiografías. |
| **Streamlit** | Interfaz web interactiva. |
| **Matplotlib / NumPy** | Visualización y cálculos numéricos. |
| **Scikit-learn** | Cálculo de pesos de clase. |

## 7. Arquitectura 
El sistema sigue una arquitectura modular dividida en cuatro capas principales: Interfaz de Usuario, Procesamiento de Imágenes, Motor de IA y Explicabilidad. Cada capa tiene una responsabilidad clara y se comunican mediante un flujo secuencial de datos.

### Componentes Principales:
### Interfaz de Usuario (Streamlit)

- Permite cargar radiografías (JPG, PNG, JPEG).
- Muestra la imagen original y la mejorada lado a lado.
- Presenta el diagnóstico, la confianza y el mapa Grad-CAM.

### Procesamiento Digital de Imágenes (OpenCV)

- Mejora automática: Aplica CLAHE (clipLimit=1.5) para mejorar el contraste local y un filtro Bilateral para reducir ruido sin perder bordes.
- Métricas de calidad: Calcula contraste (desviación estándar) y nitidez (varianza del Laplaciano).
- Evaluación de calidad: Clasifica la imagen como Reject, Warning u Optimal según umbrales empíricos.

### Motor de Inteligencia Artificial (PyTorch + TorchXRayVision)

- Modelo: DenseNet121 preentrenado en radiografías (TorchXRayVision) y fine-tuneado con datos de neumonía y tuberculosis.
- Preprocesamiento: Redimensiona a 224x224, normaliza a [0,1] y convierte a 3 canales (RGB).
- Salida: 3 probabilidades (Normal, Neumonía, Tuberculosis) mediante Softmax.

### Explicabilidad

- Genera un mapa de calor sobre la capa denseblock4.denselayer16.conv2.
- Superpone las regiones de activación sobre la radiografía para mostrar qué áreas influyeron en la decisión.

---


## 8. Construcción del Sistema
## 9. Modelo de Inteligencia Artificial


## 10. Capturas y evidencia


<img width="1339" height="949" alt="image" src="https://github.com/user-attachments/assets/dc585003-d929-4ba0-ba98-45414a4b56cb" />


<img width="1099" height="863" alt="EntrenamientoCaptura" src="https://github.com/user-attachments/assets/0cbf9081-008d-486f-bb35-3f748ad5f4e5" />


## 11. Instalación y Uso

### Requisitos
- Python 3.9 o superior.
- GPU NVIDIA (opcional, pero recomendada).

### 1. Clonar el repositorio (o descargar)
```bash
git clone https://github.com/tu_usuario/MedVision-AI.git
cd MedVision-AI
```

### 2. Crear y activar entorno virtual
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Organizar el dataset 
```bash
python organize_dataset.py
```

### 5. Entrenar el modelo (opcional, ya se proporciona uno pre-entrenado)
```bash
python train.py
```

### 6. Ejecutar la aplicación
```bash
streamlit run app.py
```
La interfaz estará disponible en http://localhost:8501.

---

### 12.Créditos
Autor: José Manuel Montalvo Espinoza

Curso: Computación Gráfica

Universidad Nacional de Ingeniería
