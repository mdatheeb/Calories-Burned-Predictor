import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Calories Burned Predictor",
                   page_icon="🔥", layout="centered")


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, 'features.pkl'))
    return model, scaler, features


model, scaler, features = load_artifacts()

st.title("Calories Burned Predictor")
st.write("Estimate calories burned during exercise")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 80, 25)
    height = st.slider("Height (cm)", 130, 210, 170)
    weight = st.slider("Weight (kg)", 35, 150, 70)

with col2:
    duration = st.slider("Exercise Duration (minutes)", 1, 60, 15)
    heart_rate = st.slider("Heart Rate (bpm)", 60, 180, 100)
    body_temp = st.slider("Body Temperature (°C)", 36.0, 42.0, 39.5, step=0.1)

if st.button("Predict Calories Burned", type="primary"):
    gender_val = 1 if gender == "Male" else 0
    input_dict = {
        'Gender': gender_val,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
    }
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction = max(prediction, 0)

    st.success(f"### Estimated Calories Burned: {prediction:.1f} kcal")

    st.caption(
        "Model: Linear Regression | Trained on exercise/physiological data | "
        f"Test R² ≈ 0.94"
    )

with st.expander("About this model"):
    st.write("""
    This app uses a **Linear Regression** model trained on exercise session data
    (gender, age, height, weight, duration, heart rate, and body temperature)
    to predict calories burned.

    Built as a portfolio project to demonstrate an end-to-end ML workflow:
    data preprocessing → feature scaling → model training/evaluation → deployment.
    """)
