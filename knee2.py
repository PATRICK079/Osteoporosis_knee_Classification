import os
import boto3
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load AWS credentials from .env file (for local testing)
load_dotenv()

# Function to download model from S3
def download_model_from_s3(bucket_name, object_name, local_file):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    if not os.path.exists(local_file):  # Check if the file already exists locally
        with st.spinner("Downloading model... This may take a while."):
            s3.download_file(bucket_name, object_name, local_file)
            st.success("Model downloaded successfully!")

# Download the CNN model from S3
bucket_name = "edosomacnnmodel"
object_name = "cnn_model.h5"
local_file = "cnn_model.h5"

download_model_from_s3(bucket_name, object_name, local_file)

# Load the pre-trained models
tabular_model = joblib.load('Logistic_knee_model.pk1')
tabular_scaler = joblib.load('Knee_scaler.pk1')

# Load the CNN model from the local file
image_model = load_model(local_file)

# Sidebar for navigation
st.sidebar.title("Navigation")
task = st.sidebar.radio(
    "Choose a Task:",
    ("Home", "Patient Data Classification", "Image Classification")
)

st.sidebar.title("Sample X-Ray Images")
st.sidebar.markdown("""
Use the following links to download sample X-ray images for testing the model:
- [Healthy Knee Image 1](https://github.com/PATRICK079/Knee_Classification/blob/main/110.jpg)
- [Healthy Knee Image 2](https://github.com/PATRICK079/Knee_Classification/blob/main/11nn.png)
- [Osteoporosis Knee Image](https://github.com/PATRICK079/Knee_Classification/blob/main/106%20(1).jpeg)
""")

st.sidebar.title("Dataset Citation")
st.sidebar.markdown("""
This project uses the [Osteoporosis Knee X-ray Dataset](https://www.kaggle.com/datasets/stevepython/osteoporosis-knee-xray-dataset/code) from Kaggle. Check out the dataset for more details.
""")

# --------------------------------
# Home Page
# --------------------------------
if task == "Home":
    st.markdown("### Welcome to the Osteoporosis Knee Prediction App")
    st.image("man-knee-pain-he-puts-260nw-2476578973.jpg.jpg", caption="Osteoporosis Knee Prediction", use_container_width=True)
    st.markdown("""
        Use the sidebar to choose a task:
        - **Patient Data Classification**: Predict knee health using patient data.
        - **Image Classification**: Classify knee health using knee X-ray images.
    """)
    st.markdown("## About App")
    st.markdown("""
   This app is designed to predict whether a patient has a healthy knee or an osteoporosis-affected knee. Leveraging cutting-edge machine learning techniques, the app offers two prediction modes:

1. **Patient Data Analysis**: Input patient data to get a prediction based on a **Logistic Regression Model**.

   • Accuracy: 96%
   
   • K-Fold Evaluation Accuracy: 91%
   
   • Ideal for quick and reliable predictions using non-imaging patient data.

2. **Knee X-ray Image Analysis**:  Upload a knee X-ray image to utilize a Convolutional Neural Network (CNN) model trained on the Osteoporosis Knee X-ray Dataset from Kaggle.
Incorporates offline image augmentation for enhanced accuracy and robustness.
""")
    
# --------------------------------
# Tabular Data Classification Page
# --------------------------------
elif task == "Patient Data Classification":
    st.title("Patient Data Classification")

    # First row of inputs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        joint_pain = st.selectbox("Joint Pain", ["No", "Yes"])
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=25)
    with col4:
        menopause_age = st.number_input("Menopause Age", min_value=0.0, max_value=100.0, step=0.1, value=50.0)

    # Second row of inputs
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        height = st.number_input("Height (meters)", min_value=0.0, max_value=3.0, step=0.01, value=1.65)
    with col6:
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, step=0.1, value=60.0)
    with col7:
        smoker = st.selectbox("Smoker", ["No", "Yes"])
    with col8:
        diabetic = st.selectbox("Diabetic", ["No", "Yes"])

    # Third row of inputs
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        hypothyroidism = st.selectbox("Hypothyroidism", ["No", "Yes"])
    with col10:
        number_of_pregnancies = st.number_input("Number of Children", min_value=0, max_value=20, step=1, value=0)
    with col11:
        seizer_disorder = st.selectbox("Seizer Disorder", ["No", "Yes"])
    with col12:
        estrogen_use = st.selectbox("Estrogen Use", ["No", "Yes"])

    # Fourth row of inputs
    col13, col14, col15, col16 = st.columns(4)
    with col13:
        history_of_fracture = st.selectbox("History of Fracture", ["No", "Yes"])
    with col14:
        dialysis = st.selectbox("Dialysis", ["No", "Yes"])
    with col15:
        family_history_of_osteoporosis = st.selectbox("Family History of Osteoporosis", ["No", "Yes"])
    with col16:
        maximum_walking_distance = st.number_input("Maximum Walking Distance (km)", min_value=0.0, step=0.1, value=1.0)

    # Fifth row of inputs
    col17, col18, col19, col20 = st.columns(4)
    with col17:
        medical_history = st.selectbox("Medical History", ["No", "Yes"])
    with col18:
        t_score_value = st.number_input("T-Score Value", step=0.1, value=0.0)
    with col19:
        z_score_value = st.number_input("Z-Score Value", step=0.1, value=0.0)
    with col20:
        bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, step=0.1, value=22.5)

    # Sixth row of inputs
    col21, _, _, _ = st.columns(4)
    with col21:
        obesity = st.selectbox("Obesity", ["No", "Yes"])

    # Prepare inputs for the model
    input_data = np.array([[
        1 if joint_pain == "Yes" else 0, 1 if gender == "Male" else 0, age, menopause_age, height, weight,
        1 if smoker == "Yes" else 0, 1 if diabetic == "Yes" else 0, 1 if hypothyroidism == "Yes" else 0,
        number_of_pregnancies, 1 if seizer_disorder == "Yes" else 0, 1 if estrogen_use == "Yes" else 0,
        1 if history_of_fracture == "Yes" else 0, 1 if dialysis == "Yes" else 0,
        1 if family_history_of_osteoporosis == "Yes" else 0, maximum_walking_distance,
        1 if medical_history == "Yes" else 0, t_score_value, z_score_value, bmi,
        1 if obesity == "Yes" else 0
    ]])

    # Only show the "Predict" button in the "Patient Data Classification" section
    if st.button("Predict"):
        try:
            scaled_input = tabular_scaler.transform(input_data)
            prediction = tabular_model.predict(scaled_input)
            raw_prediction = tabular_model.predict_proba(scaled_input)[0]  # Probabilities for both classes

            # Define classes and confidence
            classes = ["Healthy Knee Likely", "Osteoporosis Knee Likely"]
            prediction_class = classes[np.argmax(raw_prediction)]
            confidence = raw_prediction[np.argmax(raw_prediction)]
            result_color = "green" if prediction_class == "Healthy Knee Likely" else "red"

            # Display the prediction result
            st.markdown(
                f"<span style='color:{result_color}; font-weight:bold;'>Predicted Class by Logistic Regression: {prediction_class} (Confidence: {confidence:.2%})</span>",
                unsafe_allow_html=True
            )

            # Optionally, show a pie chart with prediction probabilities
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.pie(raw_prediction, labels=classes, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722'])
            ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")


# Image Classification Page
elif task == "Image Classification":
    st.title("Knee Image Classification")

    uploaded_file = st.file_uploader("Upload an Image of a Knee", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image = image.convert("RGB")
        img_array = np.array(image.resize((150, 150))) / 255.0  # Resize image and normalize
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Predict"):
            try:
                # Make prediction with the CNN model
                prediction = image_model.predict(img_array)

                # Determine the predicted class based on the model's output
                prediction_class = "Osteoporosis Knee Likely" if prediction[0] > 0.5 else "Healthy Knee Likely"
                confidence = prediction[0] if prediction[0] < 0.5 else 1 - prediction[0]

                # Display the predicted class and confidence
                st.write(f"Prediction: **{prediction_class}**")
                
                # Optional: Add a sanity check for non-knee images after prediction
                if np.max(prediction) > 0.993:  # Example threshold for uncertainty
                    st.warning("This does not appear to be a knee image. Please upload a valid knee image.")

                # Visualization of the prediction
                result_color = "green" if prediction_class == "Healthy Knee Likely" else "red"
                st.markdown(
                    f"<span style='color:{result_color}; font-weight:bold;'>Predicted Class: {prediction_class} (Confidence: {confidence:.2%})</span>",
                    unsafe_allow_html=True
                )

                # Optionally, show a pie chart with prediction probabilities
                # Define classes for pie chart
                classes = ["Healthy Knee Likely", "Osteoporosis Knee Likely"]
                prediction_probabilities = [1 - prediction[0], prediction[0]]  # Healthy Knee and Osteoporosis Knee probabilities

                # Plot pie chart
                fig, ax = plt.subplots()
                ax.pie(prediction_probabilities, labels=classes, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5722'])
                ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Image prediction failed: {e}")
