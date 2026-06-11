import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

logo = Image.open("Images/simbolo-de-la-medicina.png")

st.set_page_config(
    page_title="MedVision Assist",
    page_icon=logo,
    layout="wide"
)

col1, col2 = st.columns([1,7])

with col1:
    st.image("Images/simbolo-de-la-medicina.png", width=100)


with col2:
    st.markdown(
        "<h1 style='color:#00BFFF;'>MedVision Assist</h1>",
        unsafe_allow_html=True
    )
    st.subheader("Sistema Inteligente de Análisis de Radiografías")

uploaded_file = st.file_uploader(
    "Sube una radiografía",
    type=["png", "jpg", "jpeg"]
)

## Inicio


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)



    gaussian = cv2.GaussianBlur(gray, (5,5), 0)

    median = cv2.medianBlur(gray, 5)




    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)



    st.markdown("## Procesamiento de Imagen")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            gray,
            caption="Imagen Original",
            use_container_width=True
        )

    with col2:
        st.image(
            enhanced,
            caption="CLAHE",
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.image(
            gaussian,
            caption="Gaussian Blur",
            use_container_width=True
        )

    with col4:
        st.image(
            median,
            caption="Median Filter",
            use_container_width=True
        )




    st.markdown("## Histogramas")

    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    axes[0,0].hist(gray.ravel(), bins=256)
    axes[0,0].set_title("Original")

    axes[0,1].hist(enhanced.ravel(), bins=256)
    axes[0,1].set_title("CLAHE")

    axes[1,0].hist(gaussian.ravel(), bins=256)
    axes[1,0].set_title("Gaussian")

    axes[1,1].hist(median.ravel(), bins=256)
    axes[1,1].set_title("Median")

    plt.tight_layout()

    st.pyplot(fig)


 

    st.markdown("## Métricas de Imagen")

    mean_intensity = np.mean(gray)

    std_intensity = np.std(gray)

    contrast = gray.max() - gray.min()

    c1, c2, c3 = st.columns(3)

    c1.metric("Brillo promedio", f"{mean_intensity:.2f}")

    c2.metric("Desviación estándar", f"{std_intensity:.2f}")

    c3.metric("Contraste", f"{contrast}")



    st.markdown("## Interpretación Preliminar")

    if contrast < 100:
        st.warning(
            "La imagen presenta bajo contraste. "
            "Se recomienda mejora de intensidad."
        )

    else:
        st.success(
            "La imagen presenta un rango de contraste adecuado."
        )

    st.info(
        "El preprocessing fue aplicado correctamente. "
        "La imagen está lista para etapas posteriores "
        "de segmentación y análisis profundo."
    )