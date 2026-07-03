<div align="center">

# MEDVISION ASSIST

<br><br>


> Sistema Inteligente para el Análisis y Clasificación de Radiografías de Tórax mediante Procesamiento Digital de Imágenes y Redes Neuronales Convolucionales.


**MedVision Assist** es un proyecto desarrollado como parte del curso **Computación gráfica** de la **Facultad de Ciencias** de la **Universidad Nacional de Ingenieria**, cuyo propósito es integrar técnicas de visión por computadora e inteligencia artificial para apoyar el análisis preliminar de radiografías de tórax.

> El proyecto asume métodos clásicos de procesamiento digital de imágenes con modelos de aprendizaje profundo para mejorar la calidad visual de las radiografías y realizar una clasificación automática de enfermedades respiratorias. A lo largo de su desarrollo se abordaron aspectos relacionados con el tratamiento de imágenes médicas, la extracción de información cuantitativa, el diseño de interfaces de usuario y la aplicación de redes neuronales convolucionales mediante aprendizaje por transferencia.


</div>

<br><br>

## Sobre esta documentación

Esta documentación no solo presenta el resultado final del proyecto, sino también el proceso seguido durante su desarrollo. Cada sección explica las decisiones de diseño, las tecnologías empleadas y la justificación de las soluciones implementadas, permitiendo comprender la evolución del sistema desde su concepción hasta su estado actual.

> El objetivo es ofrecer una visión clara y estructurada del proyecto, facilitando tanto su comprensión como su futura extensión o mantenimiento.

<br><br>

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
   - Explicación del Código Fuente
       - `app.py` – Orquestación y Lógica de Decisión
       - `train.py` – Estrategia de Entrenamiento y Fine-Tuning
       - `utils/quality_utils.py` – Filtro de Admisión
       - `utils/model_utils.py` – Carga del Modelo y Preprocesamiento para IA
       - `utils/gradcam_utils.py` – Explicabilidad Visual

9. Modelo de Inteligencia Artificial
10. Capturas y evidencia
11. Ejecución del proyecto
12. Trabajo Futuro
13. Créditos

<br><br>

## 1. Problema

Las enfermedades respiratorias, como la neumonía y la tuberculosis, representan un importante problema de salud pública debido a su alta incidencia y mortalidad a nivel mundial. En este contexto, la radiografía de tórax constituye uno de los principales métodos de apoyo para la detección y evaluación de estas patologías; sin embargo, su efectividad depende en gran medida de la calidad de la imagen obtenida y de la correcta interpretación por parte del personal médico.

Uno de los principales inconvenientes es que las radiografías pueden presentar deficiencias de calidad ocasionadas por factores técnicos durante su adquisición, tales como bajo contraste, niveles inadecuados de brillo, presencia de ruido o artefactos e incluso diferencias entre equipos de radiología. Estas condiciones pueden dificultar la visualización de estructuras anatómicas importantes y reducir la confiabilidad del análisis clínico.

A estas limitaciones se suman los desafíos propios de la interpretación humana. La elevada carga de trabajo de los especialistas, la subjetividad inherente al análisis visual y la escasez de radiólogos en diversas regiones pueden incrementar la probabilidad de errores, generar diferencias en los diagnósticos y retrasar la atención de los pacientes.

Frente a esta problemática, surge la necesidad de desarrollar herramientas tecnológicas que permitan mejorar la calidad de las radiografías y proporcionar un apoyo objetivo durante su análisis. En este sentido, el proyecto **MedVision Assist** busca contribuir a esta necesidad mediante la integración de técnicas de Procesamiento Digital de Imágenes e Inteligencia Artificial, permitiendo optimizar la calidad visual de las radiografías, obtener métricas objetivas sobre la imagen y realizar una clasificación automática de enfermedades respiratorias como **Normal**, **Neumonía** y **Tuberculosis**. De esta manera, el sistema pretende servir como una herramienta de apoyo para los profesionales de la salud, facilitando el análisis preliminar de las imágenes y contribuyendo a una toma de decisiones más rápida, consistente y fundamentada.

<br><br>

## 2. Propuesta

La solución propuesta es MedVision Assist: un sistema inteligente híbrido que combina un robusto pipeline de procesamiento digital de imágenes (preprocesamiento) con un clasificador basado en aprendizaje profundo
(Deep Learning) para la identificación preliminar de neumonía, tuberculosis o normalidad en radiografías de tórax.


<br><br>

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

<br><br>

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

<br><br>

## 5. Flujo de trabajo

1. **El usuario sube una radiografía** (JPG, JPEG, PNG).
2. **El sistema evalúa la calidad original** (contraste, nitidez, entropía) y asigna un estado (`reject`, `warning`, `optimal`).
3. **Se aplica mejora automática** (CLAHE + Filtro Bilateral) y se re-evalúa la calidad.
4. **Si la calidad es óptima o aceptable**, la imagen se envía al modelo IA.
5. **El modelo DenseNet121** clasifica la imagen en **Normal**, **Neumonía** o **Tuberculosis**.
6. **Se muestra el diagnóstico** con la confianza ajustada (penalización del 20% si la calidad es `warning`).
7. **Se genera un mapa de calor Grad-CAM** para visualizar las regiones que influyeron en la decisión.

<br><br>

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

<br><br>

## 7. Arquitectura 
El sistema sigue una arquitectura modular dividida en cuatro capas principales: Interfaz de Usuario, Procesamiento de Imágenes, Motor de IA y Explicabilidad. Cada capa tiene una responsabilidad clara y se comunican mediante un flujo secuencial de datos.

### Componentes Principales:
### Interfaz (Streamlit)

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


<br><br>

## 8. Construcción del Sistema

La implementación de MedVision Assist se ha desarrollado siguiendo una arquitectura modular estrictamente desacoplada. Este enfoque permite que cada componente (interfaz, procesamiento, métricas, inteligencia artificial y explicabilidad) pueda ser probado, mejorado o reemplazado de manera independiente .

A continuación, se desglosan los módulos que conforman el núcleo del sistema, detallando su interacción, los fundamentos técnicos y explicacion de código realizado en el proyecto.

### 8.1. Interfaz(Frontend)

La capa de presentación está construida íntegramente con Streamlit, un framework de Python que permite crear aplicaciones web de datos con solo unas pocas líneas de código. La elección de Streamlit fue mas por practicidad teniendo en cuenta que no es algo profesional pero sí mas educativo. 

Características de la Interfaz:

- Carga de Imágenes: Mediante st.file_uploader, el usuario puede arrastrar o seleccionar radiografías en formatos JPG, JPEG o PNG.
- Visualización Lateral: Se emplean dos columnas (st.columns(2)) para mostrar la imagen Original y la Procesada en paralelo, facilitando la comparación visual inmediata.
- Panel de Métricas: En la barra lateral (st.sidebar) se despliegan indicadores numéricos de calidad y la interpretación automática.
- Expansores de Detalle: Componentes como st.expander permiten ocultar/mostrar los histogramas y el código de procesamiento, manteniendo la vista principal despejada.
- Gestión de Estado: Se utiliza st.session_state para conservar las imágenes y resultados entre interacciones del usuario, evitando recálculos innecesarios.

### 8.2. Procesamiento Digital de Imágenes

El pipeline de procesamiento es el corazón técnico del sistema. Su objetivo es transformar la radiografía cruda (afectada por ruido y bajo contraste) en un insumo de alta calidad, tanto para la percepción humana como para el posterior análisis de la red neuronal. Este proceso se ejecuta en el módulo `quality_utils.py` y sigue una secuencia definida

- CLAHE (clipLimit=2.0, tileGridSize=(8,8)): El límite de contraste en 2.0 evita la sobreamplificación en áreas como el mediastino o el diafragma, donde las densidades son homogéneas. La rejilla de 8x8 garantiza un equilibrio entre detalle local y coherencia global.
- Filtro Gaussiano (kernel 5x5, σ=0.8): Un kernel de 5x5 con sigma baja (0.8) proporciona un suavizado ligero, eliminando el ruido de alta frecuencia sin desdibujar las consolidaciones neumónicas pequeñas.
- Filtro de Mediana (kernel 3x3): Suficiente para eliminar el ruido "sal y pimienta" sin afectar el tamaño de las lesiones.

### 8.3. Generación y Análisis de Histogramas

- Se generan dos histogramas superpuestos: el de la imagen original y el de la procesada.
- El eje X representa los 256 niveles de intensidad (0 = negro, 255 = blanco).
- El eje Y representa la frecuencia de píxeles en cada nivel.

Interpretación Clínica Automatizada:

- Histograma concentrado en el centro: Indica bajo contraste (pérdida de diferenciación entre tejidos).
- Histograma desplazado a la izquierda (< 100): Subexposición; las estructuras oscuras (como el corazón) se funden con el fondo.
- Histograma desplazado a la derecha (> 200): Sobreexposición; los pulmones aparecen "quemados" (blancos), ocultando patologías.
- Histograma distribuido a lo largo de todo el rango: Calidad óptima; existe diferenciación entre agua, grasa, hueso y aire.



### 8.4. Métricas usadas en el proyecto

Con el objetivo de evaluar objetivamente la calidad visual de una radiografía y determinar si es adecuada para el análisis mediante inteligencia artificial, el sistema calcula tres métricas fundamentales implementadas en `quality_utils.py`. Estas métricas convierten la evaluación subjetiva de la imagen en indicadores cuantitativos, permitiendo decidir si la radiografía posee la calidad suficiente para ser procesada por el modelo de clasificación.

| **Métrica** | **Fórmula Matemática** | **Implementación** | **Interpretación Clínica** |
|-------------|------------------------|--------------------|----------------------------|
| **Contraste** | $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(I_i-\bar{I})^2}$$ | `np.std(image)` | Mide la dispersión de los niveles de intensidad de la imagen. Un valor **mayor a 65** indica una excelente diferenciación entre tejidos (aire, partes blandas y hueso), mientras que un valor **menor a 40** sugiere bajo contraste, dificultando la identificación de estructuras anatómicas y posibles lesiones. |
| **Nitidez (Sharpness)** | $$\|\nabla I\|=\sqrt{\left(\frac{\partial I}{\partial x}\right)^2+\left(\frac{\partial I}{\partial y}\right)^2}$$ | `cv2.Laplacian(image, cv2.CV_64F).var()` | Se estima mediante la **varianza del operador Laplaciano**, que cuantifica la intensidad de los bordes presentes en la imagen. Valores elevados indican contornos bien definidos (vasos pulmonares, costillas y bordes pleurales), mientras que valores bajos evidencian imágenes borrosas debido al movimiento del paciente o problemas de enfoque. |
| **Entropía** | $$H=-\sum_{i=0}^{255} p_i \log_2(p_i)$$ | `skimage.measure.shannon_entropy(image)` | Mide la cantidad de información o complejidad presente en la imagen. Radiografías con una distribución uniforme de intensidades presentan una entropía moderada, mientras que patologías como neumonía o tuberculosis incrementan la variabilidad textural y, por consiguiente, el valor de la entropía en las regiones afectadas. |

Estas tres métricas permiten realizar una evaluación integral de la calidad de la radiografía antes de su procesamiento por el modelo de inteligencia artificial. El contraste garantiza una adecuada diferenciación entre estructuras anatómicas, la nitidez verifica la presencia de detalles relevantes para el diagnóstico y la entropía estima la cantidad de información útil contenida en la imagen. La combinación de estos indicadores reduce la probabilidad de analizar imágenes deficientes, mejorando la confiabilidad de las predicciones realizadas por el sistema.


### 8.5. Interpretación Automática (Filtro de Admisión)

Esta capa de lógica, alojada en `quality_utils.py`, consume las métricas calculadas y emite un juicio semántico sobre la aptitud de la imagen. No solo informa al usuario, sino que actúa como un "filtro de admisión" para el modelo de IA.
Este filtro es crucial para evitar que el modelo de Deep Learning realice predicciones sobre insumos de baja calidad, lo que aumentaría la tasa de falsos positivos/negativos.


### 8.6 Explicación del Código Fuente

> `app.py`


### Carga de la radiografía

El usuario carga una imagen en formato **JPG**, **JPEG** o **PNG** mediante el componente `st.file_uploader()`. Posteriormente, la imagen es decodificada con **OpenCV** y convertida a escala de grises para su procesamiento.

```python
uploaded_file = st.file_uploader("Cargue una radiografía (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

---

### Evaluación automática de la calidad

Antes de realizar cualquier diagnóstico, el sistema calcula tres métricas fundamentales de calidad:

- **Contraste**
- **Nitidez (Sharpness)**
- **Entropía**

Estas métricas permiten clasificar la radiografía en tres niveles:

-  **Optimal**
-  **Warning**
-  **Reject**

```python
    cont_orig, grad_orig, ent_orig = compute_metrics(img_gray)
    level_orig = get_quality_level(cont_orig, grad_orig)
```

---

### Mejora automática de la imagen

Cuando la calidad original es insuficiente, el sistema aplica un proceso automático de mejora utilizando técnicas de procesamiento de imágenes. Posteriormente vuelve a evaluar las métricas para determinar si la imagen mejorada ofrece mejores condiciones para el análisis.

```python
    img_enhanced = apply_auto_enhancement(img_gray)
    cont_enh, grad_enh, ent_enh = compute_metrics(img_enhanced)
    level_enh = get_quality_level(cont_enh, grad_enh)
```

La versión (original o mejorada) con mayor calidad será utilizada durante la inferencia.

---

### Validación de calidad

Si incluso después del proceso de mejora la imagen continúa clasificándose como **Reject**, el sistema bloquea el diagnóstico para evitar predicciones poco confiables.

```python
    if final_level == 'reject':
        st.error("- **DIAGNÓSTICO BLOQUEADO**")
        st.error("La calidad de la imagen es insuficiente. Cargue una radiografía con mejor exposición y definición.")
        st.stop()
```

Esta validación constituye una de las principales características del proyecto, ya que evita que el modelo procese imágenes con calidad insuficiente.

---

### Preprocesamiento e inferencia

La imagen seleccionada es preprocesada para adaptarse al formato requerido por el modelo basado en **TorchXRayVision**.

Posteriormente se realiza la inferencia utilizando **PyTorch**, obteniendo las probabilidades de pertenecer a cada una de las tres clases:

- Normal
- Neumonía
- Tuberculosis

```python
    if st.button(" Ejecutar Diagnóstico", type="primary"):
        with st.spinner("Cargando modelo y procesando imagen..."):
            # Cargar modelo
            model = load_model()

            input_tensor = preprocess_for_ai(final_image)
            
            # Inferencia
            raw_probs = predict(model, input_tensor)
            
            if len(raw_probs) == 18:
                probs = map_to_3_classes(raw_probs)
            else:
                probs = raw_probs  # Ya son 3 clases
            
            class_names = ['Normal', 'Neumonía', 'Tuberculosis']
            predicted_class = class_names[np.argmax(probs)]
            confidence = np.max(probs)

            ...
```

---

### Ajuste de confianza

Si la imagen presenta una calidad **Warning**, el sistema reduce automáticamente la confianza del diagnóstico para reflejar la incertidumbre asociada a una menor calidad de entrada.

```python
            if final_level == 'warning':
                confidence_penalized = confidence * 0.8
                st.warning(f" Calidad subóptima. Confianza penalizada a {confidence_penalized:.2%}")
            else:
                confidence_penalized = confidence
                st.success(f" Confianza del modelo: {confidence_penalized:.2%}")


            ....
```

Este mecanismo evita presentar resultados con un nivel de certeza excesivo cuando la imagen no cumple completamente los estándares de calidad.

---

### Interpretabilidad mediante Grad-CAM

Para aumentar la transparencia del modelo, se genera un mapa de activación **Grad-CAM**, el cual resalta las regiones de la radiografía que influyeron en la decisión de la red neuronal.

```python
                ...
                try:
                    heatmap = generate_gradcam(model, input_tensor, np.argmax(probs))
                    st.image(heatmap, caption=f"Grad-CAM: {predicted_class}", use_column_width=True)
                except Exception as e:
                    st.warning("No se pudo generar Grad-CAM. Mostrando imagen original.")
                    st.image(final_image, caption="Imagen analizada", use_column_width=True, channels="GRAY")

                ...
```

Esta visualización facilita la interpretación del diagnóstico generado por la inteligencia artificial.

---

### Presentación de resultados

Finalmente, la aplicación muestra al usuario:

- La clase predicha.
- La confianza del modelo.
- Las probabilidades para cada enfermedad.
- El mapa Grad-CAM.
- Un mensaje recordando que el sistema constituye únicamente una herramienta de apoyo clínico y **no reemplaza el criterio médico profesional**.



<br><br>

## 9. Modelo de Inteligencia Artificial

El sistema emplea un modelo de Deep Learning basado en la arquitectura DenseNet121, entrenado mediante Transfer Learning y Fine‑Tuning para clasificar radiografías de tórax en tres categorías: Normal, Neumonía y Tuberculosis. A continuación se describe el enfoque conceptual, la estrategia de entrenamiento y las decisiones técnicas que fundamentan su funcionamiento.

### 9.1. Enfoque General: Transfer Learning desde el Dominio Radiológico
En lugar de entrenar una red neuronal desde cero (lo que requeriría millones de imágenes etiquetadas y semanas de cómputo), se parte de un modelo que ya ha sido entrenado con un volumen masivo de radiografías de tórax. Concretamente, se utiliza la implementación de DenseNet121 proporcionada por la librería TorchXRayVision, cuyos pesos fueron aprendidos a partir de más de 400.000 radiografías procedentes de conjuntos de datos clínicos de referencia (NIH ChestX‑ray14, CheXpert, MIMIC‑CXR, PadChest, entre otros).

Este modelo preentrenado ya ha internalizado características visuales fundamentales para el análisis radiológico: bordes de costillas, siluetas cardíacas, opacidades pulmonares, patrones de consolidación y cavitación, etc. Al partir de este conocimiento, la red no tiene que aprender a “ver” un pulmón desde cero; en su lugar, solo necesita ajustar sus capas finales para especializarse en las tres clases concretas que interesan en este proyecto.

### 9.2. Adaptación a 3 Clases (Fine‑Tuning)
El modelo original de TorchXRayVision fue diseñado para detectar 18 patologías de forma independiente (cada una con su propia probabilidad). Para nuestro problema, se reemplaza la cabeza de clasificación por una nueva arquitectura que consta de:

- Una capa densa de 1024 a 512 neuronas.
- Una activación ReLU (que introduce no linealidad).
- Una capa de Dropout con probabilidad 0.5, que apaga aleatoriamente la mitad de las neuronas durante el entrenamiento para evitar el sobreajuste.
- Una capa final de 512 a 3 neuronas, que corresponde a las tres clases.

La salida de esta cabeza se procesa con una función Softmax, que convierte las puntuaciones (logits) en probabilidades que suman 1, de modo que cada radiografía recibe una distribución de confianza entre las tres categorías.

### 9.3. Estrategia de Entrenamiento en Dos Fases
Para aprovechar al máximo el conocimiento preentrenado y evitar dañar las representaciones ya aprendidas, el entrenamiento se divide en dos etapas:

- Fase 1: Entrenamiento de la Cabeza (Backbone Congelado)
  
   - **Objetivo:** Enseñar al nuevo clasificador a interpretar las características extraídas por el backbone (las capas convolucionales) y mapearlas a las tres clases.
   
   - **Procedimiento:** Se congelan todos los pesos del backbone (no se actualizan) y solo se entrenan los pesos de la cabeza personalizada.
   
   - **Duración:** 5 épocas con una tasa de aprendizaje alta (1e‑3), lo que permite que la cabeza se adapte rápidamente.

- Fase 2: Fine‑Tuning (Backbone Descongelado)
   - **Objetivo:** Ajustar finamente las capas más profundas del backbone para que se especialicen en los patrones específicos de las radiografías del proyecto (consolidaciones neumónicas, cavitaciones tuberculosas, etc.).
   
   - **Procedimiento:** Se descongelan todas las capas (o las más profundas) y se entrena todo el modelo con una tasa de aprendizaje diez veces menor (1e‑4). Esto permite realizar ajustes sutiles sin destruir el conocimiento general adquirido en el preentrenamiento.
   
   - **Regularización:** Se emplea un scheduler ReduceLROnPlateau que reduce la tasa de aprendizaje si la precisión en validación se estanca, y un mecanismo de Early Stopping con paciencia de 5 épocas, que detiene el entrenamiento cuando la métrica de validación deja de mejorar, previniendo el sobreajuste.

Esta estrategia en dos fases es estándar en Transfer Learning y ha demostrado ser eficaz para adaptar modelos generalistas a tareas específicas con conjuntos de datos de tamaño moderado.

### 9.4. Manejo del Desbalance de Clases
El conjunto de datos presenta un desbalance significativo: hay muchas más imágenes de neumonía que de tuberculosis y que de normales. Si no se corrige, el modelo tendería a clasificar la mayoría de las imágenes como neumonía, porque es la clase más frecuente.

Para contrarrestar este sesgo, se utiliza una función de pérdida ponderada. Se calculan automáticamente pesos inversamente proporcionales a la frecuencia de cada clase (mediante compute_class_weight con modo 'balanced') y, adicionalmente, se incrementa manualmente el peso de la clase Normal (multiplicado por 1.8). Esta decisión se tomó después de observar que, en pruebas iniciales, muchas imágenes normales eran clasificadas erróneamente como neumonía. Al darle más importancia a la clase Normal durante el entrenamiento, el modelo se ve forzado a prestar atención a las características que distinguen un pulmón sano, reduciendo drásticamente los falsos positivos.

Los pesos finales utilizados son: Normal: 2.65, Neumonía: 0.51, Tuberculosis: 2.82.

### 9.5. Preprocesamiento Específico para la IA
El modelo fue entrenado con imágenes en formato RGB (3 canales) y con una normalización al rango [0, 1]. Por lo tanto, antes de pasar la imagen a la red, se aplica el siguiente preprocesamiento:

- Redimensionamiento a 224×224 píxeles (tamaño de entrada esperado por DenseNet121).
- Normalización dividiendo los valores de píxel entre 255.0 para llevarlos al rango [0, 1].
- Conversión a 3 canales: la imagen en escala de grises se expande a 3 canales replicando el mismo canal tres veces, simulando una imagen RGB.

Es importante destacar que la normalización utilizada no es la que emplea TorchXRayVision por defecto (que usa el rango [-1024, 1024]), sino la que se utilizó durante el entrenamiento del modelo fine‑tuneado. Esta coherencia entre el preprocesamiento de entrenamiento y el de inferencia es crucial para que las predicciones sean fiables.

### 9.6. Explicabilidad: Grad‑CAM
Uno de los aspectos más críticos en sistemas de apoyo al diagnóstico es la capacidad de explicar por qué la IA ha tomado una determinada decisión. Para ello, se implementa la técnica Grad‑CAM (Gradient‑weighted Class Activation Mapping).

Grad‑CAM utiliza los gradientes de la clase predicha con respecto a la última capa convolucional del modelo (en este caso, denseblock4.denselayer16.conv2). Estos gradientes se promedian para obtener la importancia de cada canal de activación, y se genera un mapa de calor que resalta las regiones de la imagen que más influyeron en la decisión. El mapa se superpone sobre la radiografía original, mostrando en colores cálidos (rojo, amarillo) las áreas de mayor relevancia.

Esta funcionalidad permite al médico verificar si el modelo está enfocándose en zonas anatómicamente coherentes (por ejemplo, una consolidación lobar o una cavitación) o si, por el contrario, está siendo influenciado por artefactos o ruido. Aporta así una capa de transparencia y confianza que es indispensable en entornos clínicos.

### 9.7. Resultados Clave
El modelo entrenado alcanza una precisión del 98.65 % en el conjunto de validación, con un Early Stopping que detuvo el entrenamiento en la época 15. Las pruebas con imágenes reales de las tres clases muestran un rendimiento excelente, con solo 1 error en una muestra de 15 imágenes (precisión del 93.33 % en esa prueba). La confianza mostrada en la interfaz refleja la probabilidad asignada por el modelo; valores del 100 % indican una certidumbre extrema (probabilidad > 99.995 %), lo cual es esperable cuando la imagen es muy representativa y de alta calidad.

<br><br>

## 10. Capturas y evidencia

Primer entrenamiento:

<img width="1339" height="949" alt="image" src="https://github.com/user-attachments/assets/dc585003-d929-4ba0-ba98-45414a4b56cb" />

Segundo entrenamiento:

<img width="1099" height="863" alt="EntrenamientoCaptura" src="https://github.com/user-attachments/assets/0cbf9081-008d-486f-bb35-3f748ad5f4e5" />

Programma en ejecución:

<img width="1917" height="896" alt="image" src="https://github.com/user-attachments/assets/20adeb48-e5e6-4d59-a428-d71a9dd4b61d" />


<img width="1710" height="772" alt="image" src="https://github.com/user-attachments/assets/f5948461-48ed-4a2a-83e7-1f20d296aa6c" />


<img width="1884" height="832" alt="image" src="https://github.com/user-attachments/assets/ad38a6c2-8b35-4dba-a427-4c6ec9acff60" />


<img width="1818" height="573" alt="image" src="https://github.com/user-attachments/assets/9ac0a6ac-2fac-455c-924e-8acb03526040" />


<img width="1885" height="889" alt="image" src="https://github.com/user-attachments/assets/f3e2800f-f697-40b2-9952-81ce6ee1a7ec" />


<br><br>

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

### 12. Créditos

> Autor: José Manuel Montalvo Espinoza

> Curso: Computación Gráfica

> Ciencia de la computación - Universidad Nacional de Ingeniería
