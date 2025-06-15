import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import os

# Page config
st.set_page_config(page_title="Solar Panel Defect Classifier", layout="centered")

st.title("🔍 SolarGuard: Panel Defect Detection Classifier")
st.markdown("Upload an image of a solar panel to classify its condition:")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/solar_panel_detection_MobileNetV2_model.keras")

model_path = load_model()

# Model path
# model_path = "model/solar_panel_detection_MobileNetV2_model.h5"

# Load model with error handling
# try:
#     if not os.path.exists(model_path):
#         st.error(f"🚫 Model file not found at `{model_path}`.")
#         st.stop()

#     from tensorflow.keras.applications import MobileNetV2  # Include in case of transfer learning
#     model = load_model(model_path, compile=False)
#     # Optional: include custom_objects if needed
#     # model = load_model(model_path, compile=False, custom_objects={"MobileNetV2": MobileNetV2})
# except Exception as e:
#     st.error("❌ Failed to load the model.")
#     st.exception(e)
#     st.stop()

# Class labels
class_names = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

# File uploader
uploaded_file = st.file_uploader("📤 Upload a solar panel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = image.resize((500, 500))  # Match model input size
        img_array = np.array(img_resized)
        img_preprocessed = preprocess_input(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Predict
        prediction = model.predict(img_batch)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Show results
        st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
        st.markdown(f"Confidence: **{confidence * 100:.2f}%**")

        # Display full class probabilities
        st.subheader("🔢 Class Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {prediction[0][i]*100:.2f}%")

    except Exception as e:
        st.error("❌ Error during prediction.")
        st.exception(e)
