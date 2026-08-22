# 🔥 Calories Burned Predictor

A Linear Regression project that predicts calories burned during exercise based on
gender, age, height, weight, exercise duration, heart rate, and body temperature.

## Live Demo
(i'm too lazy)

## Overview
This project walks through a full ML workflow — from raw data to a deployed,
interactive web app — using a simple and interpretable Linear Regression model.

## Dataset
Trained on exercise/physiological data (gender, age, height, weight, duration,
heart rate, body temperature → calories burned).

## Model
- **Algorithm:** Linear Regression (scikit-learn), benchmarked against Ridge Regression
- **Preprocessing:** Gender encoded numerically, features standardized with `StandardScaler`
- **Performance:** R² ≈ 0.94, RMSE ≈ 11.5 kcal on held-out test data

## Project Structure
```
calories-predictor/
├── app.py              # Streamlit app (user-facing interface)
├── linear_regression.py             # Model training & evaluation script
├── calories.csv         # Training data
├── model.pkl             # Trained Linear Regression model
├── scaler.pkl             # Fitted StandardScaler
└──features.pkl           # Feature order used by the model
```

## Run Locally
```bash
python linear_regression.py       # optional, retrains the model
streamlit run app.py
```

## Deploy (Streamlit Community Cloud — free)
1. Push this repo to GitHub (public)
2. Go to share.streamlit.io and sign in with GitHub
3. Click New app, select this repo, and set the main file to app.py
4. Click Deploy — live in a couple of minutes

## Key Takeaways
- End-to-end ML pipeline: data preprocessing → feature scaling → model training/evaluation → deployment
- Compared Linear vs Ridge Regression to check for multicollinearity effects
- Interpreted standardized coefficients to identify the strongest predictors —
  exercise duration and heart rate had the largest impact on calories burned

## Tech Stack
Python · scikit-learn · pandas · NumPy · Streamlit

## Author
Md Atheeb 
