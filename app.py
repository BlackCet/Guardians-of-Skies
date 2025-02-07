import streamlit as st
import numpy as np
import random
from PIL import Image

# Egyptian-themed UI Styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1b18; /* Dark Pharaoh-like background */
        color: #E5C100; /* Golden text */
        font-family: 'Papyrus', sans-serif;
    }
    .stButton>button {
        background-color: #E5C100; /* Gold buttons */
        color: black;
        border-radius: 10px;
    }
    .stTitle {
        font-size: 36px;
        text-align: center;
        color: #E5C100;
        font-weight: bold;
    }
    .stApp {
        background: url('images/bg_image.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Mock function to simulate model output
def mock_model(image):
    # Simulate bounding boxes and labels
    boxes = [(50, 50, 200, 200, random.choice(["Horus Guardian (Friendly)", "Anubis Raider (Enemy)"]))]
    return boxes

# Streamlit UI
st.title("🏺 GuaRdIaNs oF ThE SkIeS")
st.subheader("🔍 Decode the Skies Like an Ancient Pharaoh")

# File uploader
uploaded_file = st.file_uploader("📜 Choose an image to analyze", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Simulating Model Prediction
    results = mock_model(image)

    # Display results
    for (x, y, w, h, label) in results:
        color = "🟢" if "Friendly" in label else "🔴"
        st.write(f"{color} **{label} Aircraft Detected!**")

    if any("Enemy" in label for (_, _, _, _, label) in results):
        st.error("🚨 Anubis Raider Detected! Alert Triggered 🚨")

st.subheader("🔻 Ancient Skies: Sample Aircraft")
st.image([
    "images/sample_f16.jpg", 
    "images/sample_be200.jpg", 
    "images/sample_us2.jpg"
], caption=["🦅 Horus Guardian (F16)", "🐺 Anubis Raider (BE200)", "☀️ Ra's Falcon (US2)"], width=200)
