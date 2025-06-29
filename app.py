import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('mobilenetv2_best.h5')

model = load_keras_model()

# Define classes
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

# --- UI AND LOGIC ---

st.title('👁️ Diabetic Retinopathy Detection')
st.markdown("This application uses a Deep Learning model (MobileNetV2) to predict the stage of Diabetic Retinopathy from retinal fundus images.")

# --- SIDEBAR ---
st.sidebar.title("Choose an Image")

# Example Images
st.sidebar.markdown("### Try an Example")
example_images = {
    "No DR": "noDR1.jpg",
    "Mild": "Mild1.png",
    "Moderate": "moderate1.jpg",
    "Severe": "severe1.jpg",
    "Proliferate": "pdr1.jpg"
}
selected_example = st.sidebar.selectbox("Select an example image", list(example_images.keys()))

st.sidebar.markdown("---")

# Image Uploader
st.sidebar.markdown("### Upload Your Own")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- IMAGE PROCESSING AND PREDICTION ---

def preprocess_image(image):
    """Resizes and normalizes the image."""
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    """Runs prediction on the preprocessed image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    return prediction

def save_results(image, top_label):
    """Saves the image and logs the prediction."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/predict_{timestamp}.png"
    image.save(filename)

    log_path = "results/predict_log.csv"
    new_entry = pd.DataFrame([{"filename": filename, "predicted_label": top_label}])

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
    else:
        log_df = new_entry
    
    log_df.to_csv(log_path, index=False)
    return filename

# --- DISPLAY RESULTS ---

image_to_process = None
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    st.sidebar.success("Image uploaded successfully!")
elif selected_example:
    image_to_process = Image.open(example_images[selected_example]).convert("RGB")

if image_to_process:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image_to_process, caption='Selected Image', use_column_width=True)

    with col2:
        st.subheader("Prediction Analysis")
        with st.spinner('Classifying...'):
            prediction = predict(image_to_process)
            
            # Create a DataFrame for charting
            result_df = pd.DataFrame({
                'Stage': CLASSES,
                'Confidence': prediction * 100
            })
            result_df = result_df.sort_values('Confidence', ascending=False).reset_index(drop=True)

            # Display the top prediction
            top_prediction = result_df.iloc[0]
            st.success(f"**Diagnosis: {top_prediction['Stage']}** (Confidence: {top_prediction['Confidence']:.2f}%)")

            # Display confidence chart
            st.markdown("##### Confidence Scores")
            st.bar_chart(result_df.set_index('Stage'))

            # Save results
            saved_file = save_results(image_to_process, top_prediction['Stage'])
            st.info(f"Prediction logged. Image saved as `{saved_file}`.")
else:
    st.info("Please select an example or upload an image to begin analysis.")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app is for educational purposes only and is not a substitute for professional medical advice."
)
