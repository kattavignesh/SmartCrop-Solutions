import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Paths relative to this file
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "..", "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "..", "class_indices.json")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array

# Function to predict class
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit app
st.title("🩺 Plant Disease Prediction")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, image, class_indices)
            
            # Extract only condition (after ___)
            if "___" in prediction:
                condition = prediction.split("___")[1]
            else:
                condition = prediction
            st.success(f"Leaf Condition: {condition}")
