import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- Load model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # make sure path is correct
    return model

model = load_model()

# --- UI ---
st.title("Glaucoma Detection App 👁️")
st.write("Upload a retinal image to predict glaucoma.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# --- Image preprocessing ---
def preprocess_image(image):
    image = image.resize((224, 224))  # adjust if your model uses different size
    img_array = np.array(image)

    # If grayscale → convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    return img_array

# --- Prediction logic ---
def predict(img):
    output = model.predict(img)

    label = None
    conf = None

    try:
        # Convert to numpy
        output = np.array(output)

        # Case 1: shape (1,1)
        if output.shape == (1, 1):
            conf = float(output[0][0])
            label = "Glaucoma" if conf > 0.5 else "Normal"

        # Case 2: shape (1,2) → [normal, glaucoma]
        elif output.shape[1] == 2:
            class_idx = int(np.argmax(output[0]))
            conf = float(output[0][class_idx])
            label = "Glaucoma" if class_idx == 1 else "Normal"

        # Case 3: weird fallback
        else:
            conf = float(np.max(output))
            label = f"Class {np.argmax(output)}"

    except Exception as e:
        label = "Prediction Failed"
        conf = None
        st.error(f"Error parsing prediction: {e}")

    return label, conf

# --- Main flow ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Predicting...")

    img = preprocess_image(image)

    label, conf = predict(img)

    # --- Output ---
    st.success(f"Prediction: {label}")

    if conf is not None:
        st.info(f"Confidence: {conf:.2f}")