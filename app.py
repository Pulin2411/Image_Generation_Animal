import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Try loading model from both formats
MODEL_PATH_H5 = "optimized_cnn_animal10.h5"
MODEL_PATH_SAVED = "optimized_cnn_animal10_savedmodel"

@st.cache_resource  # caches model to avoid reloading on every interaction
def load_cnn_model():
    if os.path.exists(MODEL_PATH_H5):
        st.info("Loading model from .h5 format...")
        return tf.keras.models.load_model(MODEL_PATH_H5)
    elif os.path.exists(MODEL_PATH_SAVED):
        st.info("Loading model from SavedModel format...")
        return tf.keras.models.load_model(MODEL_PATH_SAVED)
    else:
        st.error("No model found! Upload either .h5 or SavedModel folder.")
        return None

model = load_cnn_model()

# Class labels (must match training labels order)
class_labels = ['cat', 'dog', 'horse', 'sheep', 'elephant',
                'butterfly', 'chicken', 'cow', 'squirrel', 'spider']

st.title("🐾 Animal Image Classification")
st.write("Upload an animal image and let the model classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx] * 100

    st.markdown(f"### Prediction: **{class_labels[pred_idx]}** ({confidence:.2f}% confidence)")
