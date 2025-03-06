import joblib
import numpy as np
import pandas as pd

lasso_model = joblib.load(r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models\lasso_model.pkl')
scaler = joblib.load(r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models\scaler.pkl')
expected_columns = joblib.load(r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models\feature_columns.pkl')

# Input 1
test_data1 = pd.DataFrame(np.zeros((1, len(expected_columns))), columns=expected_columns)
test_data1['Age'] = 30
test_data1['Sleep Duration (hours)'] = 7
test_data1['Physical Activity Level (minutes/day)'] = 60
test_data1['Stress Level (scale: 1-10)'] = 5
test_data1['Heart Rate (bpm)'] = 70
test_data1['Daily Steps'] = 5000
test_data1['Gender_Male'] = 1
test_data1['Occupation_Student'] = 1
test_data1['BMI Category_Obese'] = 1
test_data1['Sleep Disorder_None'] = 1
test_data1['Systolic BP'] = 120
test_data1['Diastolic BP'] = 80
test_scaled1 = scaler.transform(test_data1)
print("Prediction 1:", lasso_model.predict(test_scaled1))

# Input 2
test_data2 = pd.DataFrame(np.zeros((1, len(expected_columns))), columns=expected_columns)
test_data2['Age'] = 80
test_data2['Sleep Duration (hours)'] = 0
test_data2['Physical Activity Level (minutes/day)'] = 1
test_data2['Stress Level (scale: 1-10)'] = 1
test_data2['Heart Rate (bpm)'] = 100
test_data2['Daily Steps'] = 1000
test_data2['Gender_Male'] = 0
test_data2['Occupation_Retired'] = 1
test_data2['BMI Category_Overweight'] = 1
test_data2['Sleep Disorder_Sleep Apnea'] = 1
test_data2['Systolic BP'] = 150
test_data2['Diastolic BP'] = 100
test_scaled2 = scaler.transform(test_data2)
print("Prediction 2:", lasso_model.predict(test_scaled2))