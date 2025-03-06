import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
base_path = r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models'

def check_file_exists(file_path):
    if os.path.exists(file_path):
        return True
    else:
        st.error(f"File not found: {file_path}")
        return False

# โหลดโมเดลและไฟล์ที่เกี่ยวข้อง
knn_model_path = os.path.join(base_path, 'knn_model.pkl')
lr_model_path = os.path.join(base_path, 'lr_model.pkl')
ridge_model_path = os.path.join(base_path, 'ridge_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')
feature_columns_path = os.path.join(base_path, 'feature_columns.pkl')
feature_selector_path = os.path.join(base_path, 'feature_selector.pkl')

if not all(map(check_file_exists, [knn_model_path, lr_model_path, ridge_model_path, scaler_path, feature_columns_path, feature_selector_path])):
    st.stop()

knn_model = joblib.load(knn_model_path)
lr_model = joblib.load(lr_model_path)
ridge_model = joblib.load(ridge_model_path)
scaler = joblib.load(scaler_path)
expected_columns = joblib.load(feature_columns_path)
feature_selector = joblib.load(feature_selector_path)

def predict_sleep_quality(occupation, physical_activity, stress_level, bmi_category, blood_pressure, daily_steps, model_choice):
    try:
        sys_bp, dia_bp = map(float, blood_pressure.split('/'))
    except ValueError:
        st.error("กรุณากรอกค่าความดันโลหิตในรูปแบบที่ถูกต้อง (ตัวอย่าง: 120/80)")
        return None

    input_data = {
        'Occupation': occupation,
        'Physical Activity Level (minutes/day)': physical_activity,
        'Stress Level (scale: 1-10)': stress_level,
        'BMI Category': bmi_category,
        'Systolic BP': sys_bp,
        'Diastolic BP': dia_bp,
        'Daily Steps': daily_steps
    }

    input_df = pd.DataFrame([input_data])

    # จัดการค่า Categorical Variables
    categories = {
        'Occupation': ['Student', 'Office Worker'],
        'BMI Category': ['Underweight', 'Overweight']
    }
    
    for col, cat_values in categories.items():
        input_df[col] = pd.Categorical(input_df[col], categories=cat_values)
    
    input_encoded = pd.get_dummies(input_df, columns=['Occupation', 'BMI Category'], drop_first=True, dtype=int)

    # จัดเรียงคอลัมน์ให้ตรงกับโมเดล
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    input_selected = feature_selector.transform(input_scaled)

    model_dict = {
        "KNN Regression": knn_model,
        "Linear Regression": lr_model,
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
    occupation = st.selectbox("Occupation", ("Student", "Office Worker","Retired","Manual Labor"))
    # sleep_duration = st.number_input("Sleep Duration (Hour)", min_value=0.0, max_value=24.0, value=7.0)
    physical_activity = st.number_input("Physical Activity Level (Min/Day)", min_value=1, max_value=300, value=60)
    stress_level = st.number_input("Stress Level (Scale 1-10)", min_value=1, max_value=10, value=5)
    bmi_category = st.selectbox("BMI Category", ("Underweight","Normal", "Overweight","Obese"))
    blood_pressure = st.text_input("Blood Pressure (systolic/diastolic)", value="120/80")
    daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)  # ✅ เพิ่มให้ถูกต้อง
    model_choice = st.selectbox("Select Model", ["Ridge Regression", "Linear Regression", "KNN Regression"])

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    predicted_quality = predict_sleep_quality(
        occupation, physical_activity, stress_level, bmi_category, blood_pressure, daily_steps, model_choice
    )
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.01)
        my_bar.empty()
    
    if predicted_quality is not None:
        st.success(f"Predicted Quality of Sleep: {predicted_quality:.2f} (scale: 1-10)")
