# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import xgboost as xgb
import os

# 1. โหลดข้อมูล
file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

# 2. Data Preprocessing
# df = df.rename(columns={'Sleep Duration (hours)': 'Sleep Duration'})
df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'None')  # แก้ FutureWarning
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]
df = df[(df['Physical Activity Level (minutes/day)'] > 0) | (df['Daily Steps'] == 0)]

# ✅ **เพิ่ม Daily Steps**
selected_columns = [
    'Age', 'Occupation',  'Physical Activity Level (minutes/day)',
    'Stress Level (scale: 1-10)', 'BMI Category', 'Blood Pressure (systolic/diastolic)',
    'Daily Steps',  # ✅ เพิ่มคอลัมน์ Daily Steps
    'Quality of Sleep (scale: 1-10)'  # Target
]
df = df[selected_columns]

# One-Hot Encoding เฉพาะ Occupation และ BMI Category
df_encoded = pd.get_dummies(df, columns=['Occupation', 'BMI Category'], drop_first=True)

# แยก Systolic และ Diastolic BP
df_encoded[['Systolic BP', 'Diastolic BP']] = df_encoded['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)
df_encoded = df_encoded.drop(columns=['Blood Pressure (systolic/diastolic)'])

# 3. แยก Features & Target
X = df_encoded.drop(columns=['Quality of Sleep (scale: 1-10)'])
y = df_encoded['Quality of Sleep (scale: 1-10)']

# 4. Scaling ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. แบ่งข้อมูลเป็น Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Feature Selection - XGBoost
selector_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
selector_xgb.fit(X_train, y_train)

# ✅ อัปเดต Feature Selection ให้รวม Daily Steps
xgb_selector = SelectFromModel(selector_xgb, threshold="mean", max_features=10)
X_train_selected_xgb = xgb_selector.transform(X_train)
X_test_selected_xgb = xgb_selector.transform(X_test)

X_train_final = X_train_selected_xgb
X_test_final = X_test_selected_xgb

# 7. ฝึกโมเดล Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_final, y_train)
y_pred_lr = lr_model.predict(X_test_final)
print(f"Linear Regression -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}, R²: {r2_score(y_test, y_pred_lr):.4f}")

# 8. ฝึก Ridge Regression
ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train_final, y_train)
y_pred_ridge = ridge_model.predict(X_test_final)
print(f"Ridge Regression -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}, R²: {r2_score(y_test, y_pred_ridge):.4f}")

# 9. ฝึก KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=20)
knn_model.fit(X_train_final, y_train)
y_pred_knn = knn_model.predict(X_test_final)
print(f"KNN -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_knn)):.4f}, R²: {r2_score(y_test, y_pred_knn):.4f}")

# 10. บันทึกโมเดลและ scaler
models_dir = r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models'
os.makedirs(models_dir, exist_ok=True)

joblib.dump(lr_model, os.path.join(models_dir, 'lr_model.pkl'))
joblib.dump(ridge_model, os.path.join(models_dir, 'ridge_model.pkl'))
joblib.dump(knn_model, os.path.join(models_dir, 'knn_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'feature_columns.pkl'))
joblib.dump(xgb_selector, os.path.join(models_dir, 'feature_selector.pkl'))

print("✅ โมเดลและ Scaler ถูกบันทึกเรียบร้อยแล้ว!")
