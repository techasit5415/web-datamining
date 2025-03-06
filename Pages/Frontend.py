import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import xgboost as xgb  # Add XGBoost import
from sklearn.ensemble import RandomForestRegressor

base_path = r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models'

def check_file_exists(file_path):
    if os.path.exists(file_path):
        return True
    else:
        st.error(f"File not found: {file_path}")
        return False

# โหลดโมเดลและไฟล์ที่เกี่ยวข้อง
xgb_model_path = os.path.join(base_path, 'xgb_model.pkl')
rf_model_path = os.path.join(base_path, 'rf_model.pkl')
ridge_model_path = os.path.join(base_path, 'ridge_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')
feature_columns_path = os.path.join(base_path, 'feature_columns.pkl')
feature_selector_path = os.path.join(base_path, 'feature_selector.pkl')

if not all(map(check_file_exists, [xgb_model_path, rf_model_path, ridge_model_path, scaler_path, feature_columns_path, feature_selector_path])):
    st.stop()

xgb_model = joblib.load(xgb_model_path)
rf_model = joblib.load(rf_model_path)
ridge_model = joblib.load(ridge_model_path)
scaler = joblib.load(scaler_path)
expected_columns = joblib.load(feature_columns_path)
feature_selector = joblib.load(feature_selector_path)

def predict_sleep_quality(age, sleep_duration, physical_activity, stress_level, 
                         heart_rate, daily_steps, blood_pressure, model_choice):
    try:
        sys_bp, dia_bp = map(float, blood_pressure.split('/'))
    except ValueError:
        st.error("กรุณากรอกค่าความดันโลหิตในรูปแบบที่ถูกต้อง (ตัวอย่าง: 120/80)")
        return None

    # Create input data with the 8 specified features
    input_data = {
        'Age': age,
        'Sleep Duration (hours)': sleep_duration,
        'Physical Activity Level (minutes/day)': physical_activity,
        'Stress Level (scale: 1-10)': stress_level,
        'Heart Rate (bpm)': heart_rate,
        'Daily Steps': daily_steps,
        'Systolic BP': sys_bp,
        'Diastolic BP': dia_bp
    }

    input_df = pd.DataFrame([input_data])

    # Scale and transform input
    input_scaled = scaler.transform(input_df)
    input_selected = feature_selector.transform(input_scaled)

    # Select model
    model_dict = {
        "XGBoost": xgb_model,
        "Random Forest": rf_model,
        "Ridge Regression": ridge_model
    }

    if model_choice in model_dict:
        prediction = model_dict[model_choice].predict(input_selected)
        return prediction[0]
    else:
        st.error("กรุณาเลือกโมเดลที่ถูกต้อง")
        return None

st.title("Sleep Quality Prediction")

st.header("Enter Your Details")
with st.form(key='prediction_form'):
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
    physical_activity = st.number_input("Physical Activity Level (minutes/day)", min_value=0, max_value=300, value=60)
    stress_level = st.number_input("Stress Level (scale: 1-10)", min_value=1, max_value=10, value=5)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=5000)
    blood_pressure = st.text_input("Blood Pressure (systolic/diastolic)", value="120/80")
    model_choice = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Ridge Regression"])

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    predicted_quality = predict_sleep_quality(
        age, sleep_duration, physical_activity, stress_level,
        heart_rate, daily_steps, blood_pressure, model_choice
    )
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.01)
        my_bar.empty()
    
    if predicted_quality is not None:
        st.success(f"Predicted Quality of Sleep: {predicted_quality:.2f} (scale: 1-10)")
