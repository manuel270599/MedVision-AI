import streamlit as st
import cv2
import numpy as np
import torch
from utils.quality_utils import compute_metrics, get_quality_level, apply_auto_enhancement
from utils.model_utils import load_model, preprocess_for_ai, predict, map_to_3_classes, DEVICE
from utils.gradcam_utils import generate_gradcam

# Configuración de página
st.set_page_config(page_title="MedVision Assist", layout="wide", page_icon="Images/simbolo-de-la-medicina.png")
st.title(" MedVision Assist")
st.markdown("#### Sistema Inteligente para el Análisis y Clasificación de Radiografías de Tórax")

st.sidebar.markdown("---")
st.sidebar.warning("- **Uso exclusivamente educativo**\n\nEste sistema es un prototipo de apoyo. No reemplaza el juicio clínico. Consulte siempre a un profesional de la salud.")
st.sidebar.markdown("---")
st.sidebar.info(f"Dispositivo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
st.sidebar.caption("MedVision Assist v3.0 - PyTorch + TorchXRayVision")

# Carga de imagen
uploaded_file = st.file_uploader("Cargue una radiografía (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # calidad original ---
    cont_orig, grad_orig, ent_orig = compute_metrics(img_gray)
    level_orig = get_quality_level(cont_orig, grad_orig)
    
    # mejora_automatica ---
    img_enhanced = apply_auto_enhancement(img_gray)
    cont_enh, grad_enh, ent_enh = compute_metrics(img_enhanced)
    level_enh = get_quality_level(cont_enh, grad_enh)
    
    # --- Decisión sobre qué imagen usar ---
    use_enhanced = False
    final_level = level_orig
    if level_orig == 'reject' and (level_enh == 'warning' or level_enh == 'optimal'):
        use_enhanced = True
        final_level = level_enh
    elif level_orig == 'warning' and level_enh == 'optimal':
        use_enhanced = True
        final_level = 'optimal'
    
    final_image = img_enhanced if use_enhanced else img_gray
    
    # --- Mostrar imágenes lado a lado ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("- Original -")
        st.image(img_gray, use_column_width=True, channels="GRAY")
        st.metric("Contraste", f"{cont_orig:.2f}")
        st.metric("Nitidez", f"{grad_orig:.2f}")
        estado_orig = "- Óptimo" if level_orig=='optimal' else ("- Advertencia" if level_orig=='warning' else "- Rechazado")
        st.metric("Estado", estado_orig)
    with col2:
        st.subheader("- Mejorada (Automática)")
        st.image(img_enhanced, use_column_width=True, channels="GRAY")
        st.metric("Contraste", f"{cont_enh:.2f}")
        st.metric("Nitidez", f"{grad_enh:.2f}")
        estado_enh = "- Óptimo" if level_enh=='optimal' else ("- Advertencia" if level_enh=='warning' else "- Rechazado")
        st.metric("Estado", estado_enh)
    
    if use_enhanced:
        st.info(" Mejora automática aplicada. La imagen mejorada se usará para el diagnóstico.")
    else:
        st.success(" La imagen original cumple con los estándares de calidad.")
    
    st.markdown("---")
    
    # --- Bloqueo por calidad ---
    if final_level == 'reject':
        st.error("- **DIAGNÓSTICO BLOQUEADO**")
        st.error("La calidad de la imagen es insuficiente. Cargue una radiografía con mejor exposición y definición.")
        st.stop()
    
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
            
            if final_level == 'warning':
                confidence_penalized = confidence * 0.8
                st.warning(f" Calidad subóptima. Confianza penalizada a {confidence_penalized:.2%}")
            else:
                confidence_penalized = confidence
                st.success(f" Confianza del modelo: {confidence_penalized:.2%}")
            
            # --- Mostrar resultados ---
            st.markdown("---")
            st.subheader(" Resultado del Análisis")
            
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                st.markdown(f"###  **{predicted_class}**")
                st.markdown(f"**Confianza ajustada:** {confidence_penalized:.2%}")
                prob_df = {class_names[i]: probs[i] for i in range(3)}
                st.bar_chart(prob_df)
            with col_r2:
                # Grad-CAM
                try:
                    heatmap = generate_gradcam(model, input_tensor, np.argmax(probs))
                    st.image(heatmap, caption=f"Grad-CAM: {predicted_class}", use_column_width=True)
                except Exception as e:
                    st.warning("No se pudo generar Grad-CAM. Mostrando imagen original.")
                    st.image(final_image, caption="Imagen analizada", use_column_width=True, channels="GRAY")
            
            #mensaje final
            st.warning(" **Recuerde:** Este resultado es una ayuda preliminar. La decisión clínica final debe ser tomada por un profesional de la salud.")

else:
    st.info("+ Cargue una radiografía para comenzar el análisis.") 