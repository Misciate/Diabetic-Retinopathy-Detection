# Diabetic Retinopathy Detection

This project provides a web-based application for detecting Diabetic Retinopathy (DR) from retinal fundus images. It uses a deep learning model built on the **MobileNetV2** architecture, trained on a [Kaggle dataset](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized), to classify images into five stages of DR.

---

## ✨ Features

- **User-Friendly Interface**: A simple and intuitive web UI built with Streamlit.
- **Multiple Input Options**:
  - **Upload Your Own Image**: Users can upload images in `jpg`, `jpeg`, or `png` format.
  - **Example Images**: Pre-loaded examples for each DR stage are available for quick testing.
- **Clear Results**: Displays the final diagnosis with a confidence score and a bar chart showing the model's confidence for all stages.
- **Prediction Logging**: Automatically saves the uploaded image and prediction results for future reference.

---

## 🚀 How to Use the Application

### 1. Setup the Environment

First, clone the repository and create a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### 2. Run the App

Once the environment is set up, run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

### 3. Using the Interface

1.  **Select an Image**:
    -   Use the **"Try an Example"** dropdown in the sidebar to choose a sample image.
    -   Or, upload your own retinal image using the **"Upload Your Own"** file uploader.
2.  **View the Analysis**:
    -   The application will automatically process the image and display the results.
    -   The input image is shown on the left.
    -   The prediction analysis, including the final diagnosis and confidence scores, is shown on the right.
3.  **Check the Logs**:
    -   The uploaded image and the top prediction label are saved in the `results` directory.
    -   A log of all predictions is maintained in `results/predict_log.csv`.

---

## 🩺 Prediction Stages

The model classifies images into one of the following five stages:

1.  **No DR**: No signs of diabetic retinopathy.
2.  **Mild**: Mild non-proliferative diabetic retinopathy (NPDR).
3.  **Moderate**: Moderate NPDR.
4.  **Severe**: Severe NPDR.
5.  **Proliferate**: Proliferative diabetic retinopathy (PDR), the most advanced stage.

---

> **Disclaimer**: This application is for educational and demonstrational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for any health concerns.
