import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/dog_breed_cnn.h5")  # path to your saved model
    return model

model = load_model()

# Define your dog breed labels (update with your dataset classes)
CLASS_NAMES = ['Beagle', 'Bulldog', 'German Shepherd', 'Golden Retriever', 'Poodle']

# Prediction function
def predict(image: Image.Image):
    img = image.resize((224,224))   # change size to match your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)
    return predicted_class, confidence

# Streamlit UI
st.title("🐶 Dog Breed Prediction App")
st.write("Upload a dog image and the model will predict its breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("⏳ Classifying...")
    breed, confidence = predict(image)
    st.success(f"Predicted Breed: **{breed}** with {confidence}% confidence ✅")
