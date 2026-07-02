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
4. Tecnologías Utilizadas
5. Arquitectura General
6. Construcción del Sistema
   - Interfaz
   - Procesamiento Digital de Imágenes
   - Histogramas
   - Métricas
   - Interpretación Automática
7. Modelo de Inteligencia Artificial
8. Flujo Completo del Sistema
9. Resultados
10. Trabajo Futuro
11. Créditos



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
