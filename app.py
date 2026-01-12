import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Load the trained CNN model
model = load_model("terrain_classifier.h5")
class_names = ['Desert 🏜️', 'Forest 🌳', 'Mountain ⛰️', 'Plains 🌾']

# Load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

hero_anim = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_touohxv0.json")
loading_anim = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_usmfx6bp.json")
terrain_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_5gY3X6.json")

# Image preprocessing
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Page settings
st.set_page_config(page_title="Terrain Classifier", page_icon="🌍", layout="centered")

# ------------------- THEME SWITCHER -------------------
st.sidebar.header("⚙️ Appearance Settings")
mode = st.sidebar.radio("🌗 Mode", ("Light Mode", "Dark Mode"))
theme = st.sidebar.radio("🎨 Background Theme", ("Gradient Blue", "Gradient Purple", "Abstract Pattern"))

# Base mode (light/dark)
if mode == "Dark Mode":
    base_colors = """
        body, .stApp {
            color: #f0f0f0;
        }
    """
else:  # Light Mode
    base_colors = """
        body, .stApp {
            color: #222222;
        }
    """

# Background theme
if theme == "Gradient Blue":
    background_css = """
        .stApp {
            background: linear-gradient(135deg, #141e30, #243b55);
        }
    """
elif theme == "Gradient Purple":
    background_css = """
        .stApp {
            background: linear-gradient(135deg, #654ea3, #eaafc8);
        }
    """
elif theme == "Abstract Pattern":
    background_css = """
        .stApp {
            background-image: url("https://www.toptal.com/designers/subtlepatterns/patterns/memphis-mini.png");
            background-size: cover;
        }
    """

# ------------------- CUSTOM CSS -------------------
st.markdown(f"""
    <style>
    {base_colors}
    {background_css}

    /* Center content container */
    .block-container {{
        max-width: 800px;
        margin: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    /* Gradient header */
    .main-header {{
        background: rgba(0,0,0,0.5);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }}
    .main-header h1 {{
        color: #1abc9c;
        font-size: 46px;
        font-weight: 800;
        margin: 0;
    }}
    .main-header p {{
        color: #e0e0e0;
        font-size: 18px;
        margin-top: 8px;
    }}

    /* Instruction box */
    .instruction-box {{
        background-color: rgba(30, 30, 30, 0.7);
        padding: 18px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 16px;
        color: #ccc;
    }}

    /* Footer */
    .footer {{ text-align: center; font-size: 13px; color: #aaa; margin-top: 30px; }}
    .stButton>button {{ border-radius: 8px; }}
    </style>
""", unsafe_allow_html=True)

# ------------------- UI HEADER -------------------
st.markdown("""
<div class='main-header'>
    <h1>🧭 Geospatial Terrain Recognition</h1>
    <p>AI-powered system to classify terrains as Desert, Forest, Mountain, or Plains 🌍</p>
</div>
""", unsafe_allow_html=True)

# Hero animation
if hero_anim:
    st_lottie(hero_anim, height=250, key="hero")

# Instructions
st.markdown("""
<div class='instruction-box'>
    📌 <b>How to Use:</b><br>
    1️⃣ Upload a terrain image (JPG/PNG)<br>
    2️⃣ Confirm it is a valid terrain<br>
    3️⃣ View instant predictions with confidence<br>
</div>
""", unsafe_allow_html=True)

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader("📷 Upload your terrain image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.checkbox("✅ Confirm this is a terrain image"):
        if loading_anim:
            st_lottie(loading_anim, height=120, key="loading")
        else:
            st.info("🔄 Classifying...")

        st.markdown("### 🌀 Classifying...")

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        confidence = float(np.max(prediction))
        predicted_class = class_names[np.argmax(prediction)]

        st.progress(int(confidence * 100))

        if confidence < 0.8:
            st.error(f"⚠️ Model is unsure about this image.\nConfidence: {confidence*100:.2f}%")
        else:
            st.success(f"🧭 Predicted Terrain: **{predicted_class}**")
            st.metric(label="📊 Confidence", value=f"{confidence * 100:.2f}%")

            if terrain_anim:
                st_lottie(terrain_anim, height=220, key="terrain")

            result_text = f"Prediction: {predicted_class}\nConfidence: {confidence * 100:.2f}%"
            st.download_button("📥 Download Prediction", result_text, file_name="terrain_result.txt")

    else:
        st.info("☝️ Please confirm the image is valid before classifying.")
else:
    st.info("⬆️ Upload an image to get started.")

# ------------------- FOOTER -------------------
st.markdown("<div class='footer'>✨ Project by <b style='color:#1abc9c;'>Naman Ajat & Ishaan</b> • Powered by Streamlit & TensorFlow</div>", unsafe_allow_html=True)
