from pathlib import Path

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "tomato_model.keras"
if not model_path.exists():
    model_path = base_dir.parent / "models" / "tomato_model.keras"

model = keras.models.load_model(model_path)

# Class names (same order as training)
class_names = [
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy"
]

st.title("Tomato Leaf Disease Classifier 🌿")
st.write("Upload a tomato leaf image to predict its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width="stretch")

    # Preprocess
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    index = np.argmax(predictions[0])
    confidence = predictions[0][index]
    top_indices = np.argsort(predictions[0])[-3:][::-1]

    st.subheader(f"Prediction: **{class_names[index]}**")
    st.write(f"Confidence: {confidence:.2f}")
    st.write("Top 3 probabilities:")
    for i in top_indices:
        st.write(f"- {class_names[i]}: {predictions[0][i]:.4f}")