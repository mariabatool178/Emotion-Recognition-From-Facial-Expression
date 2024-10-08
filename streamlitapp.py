import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')  # Use SavedModel path if necessary

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit UI
st.title('Emotion Recognition from Facial Expression')
st.write("Upload an image to predict the emotion!")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Preprocess the image (resize to 48x48, convert to grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 48, 48, 1))
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the emotion
    prediction = model.predict(reshaped_image)
    emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[emotion_index]
    
    st.write(f"Predicted Emotion: {predicted_emotion}")
