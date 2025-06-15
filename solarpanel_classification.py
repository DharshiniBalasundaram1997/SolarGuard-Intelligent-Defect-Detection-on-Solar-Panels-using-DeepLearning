import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load model
model = load_model("model/solar_panel_detection_MobileNetV2_model.h5")

# Class labels
class_names = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

# Streamlit UI
st.set_page_config(page_title="Solar Panel Defect Classifier", layout="centered")

st.title("🔍 SolarGuard: Panel Defect Detection Classifier")

st.markdown("Upload an image of a solar panel to classify its condition:")

uploaded_file = st.file_uploader("📤 Upload a solar panel image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width =True)

    # Preprocess
    img_resized = image.resize((500, 500)) #Resizing the uploaded image to a fixed size of 500x500. This matches with target_size used in ImageDataGenerator
    img_array = np.array(img_resized) #Convert the PIL imageinto a NumPy array.Converting to RBG pixel values (500, 500, 3)
    img_preprocessed = preprocess_input(img_array) #Scales pixel values from [0, 255] to the range [-1, 1]
    img_batch = np.expand_dims(img_preprocessed, axis=0) # Keras models expect input in batch format: batch_size x height x width x channels. (1, 500, 500, 3) instead of (500, 500, 3).

    # Predict
    prediction = model.predict(img_batch) #(1, 500, 500, 3) -> eg: predicted o/p: prediction = [[0.01, 0.87, 0.02, 0.05, 0.03, 0.02]] => (1, N_CLASSES)
    predicted_class = class_names[np.argmax(prediction)] #np.argmax(prediction) finds the index of the highest probability,     class_names[...] maps that index to the actual class label   np.argmax(...) =Dusty[1]  1 →  'Dusty'
    confidence = np.max(prediction) #highest probability score Eg: np.max(prediction) = 0.87 → 87% confidence.

    # Show results
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")

    # Optional: display full confidence scores
    st.subheader("🔢 Class Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {prediction[0][i]*100:.2f}%")   #prediction[0] = [0.01, 0.87, 0.02, 0.05, 0.03, 0.02]      prediction[0][1] = 0.87  # Dusty

        
