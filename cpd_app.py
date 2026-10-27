import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# (man)Handle import error in case gradcam.py is missing locally
try:
    from gradcam import get_gradcam_heatmap
except ImportError:
    def get_gradcam_heatmap(model, img_array, layer_name):
        return np.zeros((224, 224))

# Page config
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# UI
st.markdown(f"""
<style>
/* 1. Reset & scrollbar - Hide */
* {{ user-select: none; }}
::-webkit-scrollbar {{ display: none; }}

/* 2. Bg: Bloodstream gradient */
.stApp {{
    /* Deep red to black gradient simulating depth */
    background: radial-gradient(circle at 50% 50%, #4a0000 0%, #2a0000 40%, #000000 100%);
    background-attachment: fixed;
    background-size: cover;
    overflow: hidden; /* Prevent scrollbars from moving elements */
}}

/* 3. RBC (CSS ONLY) */
.blood-cell {{
    position: fixed;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #ff5e5e, #800000);
    box-shadow: 
        inset 10px 10px 20px rgba(0,0,0,0.5),
        inset -5px -5px 15px rgba(255,255,255,0.2),
        5px 5px 15px rgba(0,0,0,0.3);
    opacity: 0.35;
    z-index: 1 !important; /* Stays below the block-container (z-index: 10) */
    pointer-events: none;
    animation: rushDiagonal linear infinite;
}}

/* 4. Diag rush animate */
@keyframes rushDiagonal {{
    0% {{
        transform: translate(-100px, -100px) rotate(0deg) scale(0.8);
        opacity: 0;
    }}
    10% {{ opacity: 0.8; }}
    90% {{ opacity: 0.8; }}
    100% {{
        /* Move way past the bottom right corner */
        transform: translate(120vw, 120vh) rotate(360deg) scale(1.2);
        opacity: 0;
    }}
}}

/* 5. "GLASSMORPHISM CONTAINER" */
.block-container {{
    position: relative !important;
    z-index: 10 !important; /* Forces all cards, titles, and images in front of background elements */
    background: rgba(0, 0, 0, 0.7) !important;
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 2.5rem 3rem !important;
    border: 1px solid rgba(255, 77, 77, 0.25) !important;
    max-width: 950px;
    margin: auto;
}}

/* 6. Text & widget STYLIN(💅yaasssss */
h1, h2, h3, h4, h5, h6, label, .stMarkdown p, .stText, .stFileUploader label {{ 
    color: #ffebee !important; 
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
}}

/* Input/Widget text adjustments */
.stFileUploader div {{ color: white !important; }}
div[data-baseweb="file-uploader"] {{
    border: 1px dashed #ff4d4d;
}}

/* Buttons */
button {{
    background-color: rgba(139, 0, 0, 0.4) !important;
    color: white !important;
    border: 1px solid #ff4d4d !important;
    transition: all 0.3s;
}}
button:hover {{
    background-color: #ff4d4d !important;
    color: black !important;
    box-shadow: 0 0 15px #ff4d4d;
}}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# u thought i was done with UI? *prceeds to add some more UI* hehe
html_cells = ""
for i in range(20):
    # Random properties
    left_pos = random.randint(-10, 90)
    top_pos = random.randint(-10, 20)
    size = random.randint(40, 120)
    duration = random.uniform(5, 12)
    delay = random.uniform(0, 10)

    html_cells += f"""
    <div class="blood-cell" style="
        left: {left_pos}%; 
        top: {top_pos}%; 
        width: {size}px; 
        height: {size}px; 
        animation-duration: {duration}s; 
        animation-delay: -{delay}s;">
    </div>
    """

st.markdown(html_cells, unsafe_allow_html=True)


# main app
col1, col2 = st.columns([1, 8])
with col1:
    st.write("")  # Spacer
with col2:
    st.title("🫁 Chest X-Ray Pneumonia Detection")
    st.markdown("### 🧬 AI-Powered Diagnostics")

# Dhisklaimer
with st.expander("⚠️ Medical Disclaimer"):
    st.warning(
        """
        This application is for **educational purposes only**.
        It is **NOT a medical diagnostic tool**.
        Always consult a **certified medical professional**.
        """
    )


# loading model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.keras")


try:
    model = load_model()
except OSError:
    st.error("🚨 Model file not found. Please ensure 'pneumonia_model.keras' is in the directory.")
    st.stop()

LAST_CONV_LAYER = "conv5_block16_concat"

# uploading file
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

show_gradcam = st.toggle("🔥 Show Grad-CAM Heatmap")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction part
    prediction = model.predict(img_array)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Dynamic neon color for result text
    color_code = "#ff3333" if prediction > 0.5 else "#33ff33"  # Neon Red vs Neon Green
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.5); padding: 20px; border-radius: 15px; border: 1px solid {color_code}; text-align: center;">
        <h2 style="margin:0; color: white;">Result: <span style='color:{color_code}; font-weight: bold;'>{label}</span></h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")  # Spacer


    st.markdown("### 📊 Model Confidence")
    st.progress(float(confidence))
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    # grad-cam 
    if show_gradcam:
        raw_heatmap = get_gradcam_heatmap(
            model=model,
            img_array=img_array,
            layer_name=LAST_CONV_LAYER
        )

        # Smoothly resize heatmap to match original uploaded image resolution
        heatmap_tensor = np.expand_dims(raw_heatmap, axis=-1)
        heatmap_resized = tf.image.resize(
            heatmap_tensor, 
            (image.size[1], image.size[0]), 
            method='bilinear'
        ).numpy().squeeze()

        # Display side-by-side with clear, bold titles
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown(
                """
                <div style='text-align: center; margin-bottom: 12px;'>
                    <h3 style='color: #ffffff !important; font-size: 1.35rem; font-weight: 700; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,0.9);'>
                        Original Radiograph
                    </h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.image(image, use_container_width=True)

        with col_img2:
            st.markdown(
                """
                <div style='text-align: center; margin-bottom: 12px;'>
                    <h3 style='color: #ffffff !important; font-size: 1.35rem; font-weight: 700; margin: 0; text-shadow: 0 2px 8px rgba(0,0,0,0.9);'>
                        Diagnostic Grad-CAM Overlay
                    </h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

            # 1. Base radiograph
            ax.imshow(image)
            
            # 2. Smooth bilinear heatmap overlay
            ax.imshow(
                heatmap_resized, 
                cmap="jet", 
                alpha=0.45, 
                interpolation="bilinear"
            )
            ax.axis("off")
            plt.tight_layout(pad=0)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    else:
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
st.caption("🔓 Free public Streamlit Cloud deployment ready")
