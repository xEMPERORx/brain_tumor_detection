import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the models
cnn_model = load_model('brain_tumor_cnn.h5')
resnet_model = load_model('brain_tumor_resnet50.h5')

# Define the categories with proper naming conventions
categories = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to preprocess the image
def preprocess_image(image, target_size):
    image = load_img(image, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app
st.title("Brain Tumor Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Select model
    model_option = st.selectbox("Select Model", ("CNN", "ResNet"))

    if model_option == "CNN":
        model = cnn_model
        target_size = (150, 150)
    else:
        model = resnet_model
        target_size = (224, 224)

    # Button to classify
    if st.button("Classify"):
        st.write("Classifying...")
        # Preprocess the image
        image = preprocess_image(uploaded_file, target_size)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = categories[np.argmax(prediction)]

        st.write(f"Prediction: {predicted_class}")