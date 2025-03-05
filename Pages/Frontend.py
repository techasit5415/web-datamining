import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

base_path = r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models'

def check_file_exists(file_path):
    if os.path.exists(file_path):
        return True
    else:
        st.error(f"File not found: {file_path}")
        return False

knn_model_path = os.path.join(base_path, 'knn_model.pkl')
lr_model_path = os.path.join(base_path, 'lr_model.pkl')  # เพิ่ม RF Model
ridge_model_path = os.path.join(base_path, 'ridge_model.pkl')  # เพิ่ม ridge_model
scaler_path = os.path.join(base_path, 'scaler.pkl')
feature_columns_path = os.path.join(base_path, 'feature_columns.pkl')

knn_model = None
lr_model = None  # เพิ่มตัวแปร lr_model
ridge_model = None  # เพิ่มตัวแปร ridge_model
scaler = None
expected_columns = None

try:
    if not (check_file_exists(knn_model_path) and check_file_exists(lr_model_path) and check_file_exists(ridge_model_path)and 
            check_file_exists(scaler_path) and check_file_exists(feature_columns_path)):
        st.error("One or more required files are missing. Please check the file paths.")
        st.stop()

    knn_model = joblib.load(knn_model_path)
    lr_model = joblib.load(lr_model_path)  # โหลด lf_model
    ridge_model = joblib.load(ridge_model_path)  # โหลด ridge_model
    scaler = joblib.load(scaler_path)
    expected_columns = joblib.load(feature_columns_path)
    
    # st.write("Expected Columns:", list(expected_columns))
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

def predict_sleep_quality(gender, age, occupation, sleep_duration, physical_activity, stress_level, 
                         bmi_category, blood_pressure, heart_rate, daily_steps, sleep_disorder, model_choice):
    
    input_data = {
        'Gender': gender,
        'Age': age,
        'Occupation': occupation,
        'Sleep Duration (hours)': sleep_duration,
        'Physical Activity Level (minutes/day)': physical_activity,
        'Stress Level (scale: 1-10)': stress_level,
        'BMI Category': bmi_category,
        'Blood Pressure (systolic/diastolic)': blood_pressure,
        'Heart Rate (bpm)': heart_rate,
        'Daily Steps': daily_steps,
        'Sleep Disorder': sleep_disorder
    }
    
    input_df = pd.DataFrame([input_data])
    
    categories = {
        'Gender': ['Female', 'Male'],
        'Occupation': ['Manual Labor', 'Retired', 'Office Worker', 'Student'],
        'BMI Category': ['Normal', 'Underweight', 'Overweight', 'Obese'],
        'Sleep Disorder': ['None', 'Insomnia', 'Sleep Apnea']
    }
    
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_cols:
        input_df[col] = pd.Categorical(input_df[col], categories=categories[col])
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=int)
    
    input_encoded[['Systolic BP', 'Diastolic BP']] = input_encoded['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)
    input_encoded = input_encoded.drop(columns=['Blood Pressure (systolic/diastolic)'])
    
    input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # st.write("Columns after encoding:", list(input_encoded.columns))
    # st.write("Input Encoded:", input_encoded.to_dict(orient='records'))
    
    input_scaled = scaler.transform(input_encoded)
    # st.write("Input Scaled:", input_scaled.tolist())
    
    if model_choice == "KNN Regression":
        prediction = knn_model.predict(input_scaled)
    elif model_choice == "Linear Regression":
        prediction = lr_model.predict(input_scaled)  # เปลี่ยนจาก knn_model เป็น lf_model
    elif model_choice == "Ridge Regression":
        prediction = ridge_model.predict(input_scaled)
    else:
        st.error("Invalid model choice. Please select a valid model.")
        return None
    
    return prediction[0]

st.title("Sleep Quality Prediction")

st.header("Enter Your Details")
with st.form(key='prediction_form'):
    gender = st.selectbox("Gender", ("Male", "Female"))
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    occupation = st.selectbox("Occupation", ("Manual Labor", "Retired", "Office Worker", "Student"))
    sleep_duration = st.number_input("Sleep Duration (Hour)", min_value=0.0, max_value=24.0, value=7.0)
    physical_activity = st.number_input("Physical Activity Level (Min/Day)", min_value=1, max_value=300, value=60)
    stress_level = st.number_input("Stress Level (Scale 1-10)", min_value=1, max_value=10, value=5)
    bmi_category = st.selectbox("BMI Category", ("Underweight", "Normal", "Overweight", "Obese"))
    blood_pressure = st.text_input("Blood Pressure (systolic/diastolic)", value="120/80")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=1, max_value=300, value=70)
    daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)
    sleep_disorder = st.selectbox("Sleep Disorder", ("None", "Insomnia", "Sleep Apnea"))
    model_choice = st.selectbox("Select Model", ["Ridge Regression","Linear Regression", "KNN Regression"])  # เพิ่มตัวเลือก RF

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    try:
        sys_bp, dia_bp = blood_pressure.split('/')
        float(sys_bp), float(dia_bp)
    except:
        st.error("Please enter Blood Pressure in the correct format (e.g., '120/80')")
    else:
        predicted_quality = predict_sleep_quality(
            gender, age, occupation, sleep_duration, physical_activity, stress_level, 
            bmi_category, blood_pressure, heart_rate, daily_steps, sleep_disorder, model_choice
        )
        
        if predicted_quality is not None:
            st.success(f"Predicted Quality of Sleep: {predicted_quality:.2f} (scale: 1-10)")
