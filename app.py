import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("optimized_cnn_animal10.h5")

# Class labels (must match training labels order)
class_labels = ['cat', 'dog', 'horse', 'sheep', 'elephant', 
                'butterfly', 'chicken', 'cow', 'squirrel', 'spider']

st.title("🐾 Animal Image Classification")
st.write("Upload an animal image and let the model classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
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
