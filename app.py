import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# Load the trained model
model = load_model('mobilenetv2_best.h5')

# Define classes
classes = ['No diabetic retinopathy', 'Mild diabetic retinopathy', 'Moderate diabetic retinopathy', 'Severe diabetic retinopathy', 'Proliferate diabetic retinopathy']

# Create results directory if not exists
os.makedirs("results", exist_ok=True)

st.title('Diabetic Retinopathy Detection')

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.
    return np.expand_dims(image, axis=0)

# Predict function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    return prediction

# Streamlit interface
st.sidebar.title('Upload Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

    st.write('Classifying...')

    prediction = predict(image)
    prediction_percent = (prediction * 100).round(2)
    sorted_indices = np.argsort(-prediction)  # Sort descending

    st.subheader("Prediction Results:")
    for i, idx in enumerate(sorted_indices):
        label = classes[idx]
        prob = prediction_percent[idx]
        if i == 0:
            st.markdown(f"**{label}: {prob:.2f}%**")
        else:
            st.markdown(f"{label}: {prob:.2f}%")

    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/predict_{timestamp}.png"
    image.save(filename)

    # Save log: only filename and top label
    top_label = classes[sorted_indices[0]]
    log_path = "results/predict_log.csv"
    new_entry = pd.DataFrame([{
        "filename": filename,
        "predicted_label": top_label
    }])

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
    else:
        log_df = new_entry

    log_df.to_csv(log_path, index=False)
    st.success(f"Top prediction saved to `results/predict_log.csv` and image saved as `{filename}`.")
