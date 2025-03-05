# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import xgboost as xgb

file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

# 2. Data Preprocessing
df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'None')  # แก้ FutureWarning
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]
df = df[(df['Physical Activity Level (minutes/day)'] > 0) | (df['Daily Steps'] == 0)]
df = df.drop(columns=['Person ID'])

df_encoded = pd.get_dummies(df, columns=['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'], drop_first=True)
df_encoded[['Systolic BP', 'Diastolic BP']] = df_encoded['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)
df_encoded = df_encoded.drop(columns=['Blood Pressure (systolic/diastolic)'])

X = df_encoded.drop(columns=['Quality of Sleep (scale: 1-10)'])
y = df_encoded['Quality of Sleep (scale: 1-10)']

# 3. Scaling ข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. แบ่งข้อมูลเป็น Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature Selection - XGBoost
selector_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
selector_xgb.fit(X_train, y_train)
importance_xgb = selector_xgb.feature_importances_

# สำหรับ XGBoost
xgb_selector = SelectFromModel(selector_xgb, threshold="mean", max_features=10)  # เลือกฟีเจอร์ที่สำคัญที่สุด 10 ตัว
X_train_selected_xgb = xgb_selector.transform(X_train)
X_test_selected_xgb = xgb_selector.transform(X_test)


X_train_final = X_train_selected_xgb
X_test_final = X_test_selected_xgb


# 5. ฝึก KNN Regressor
n_neighbors_value = 17  # ปรับได้ตามต้องการ
knn_model = KNeighborsRegressor(n_neighbors=n_neighbors_value)
knn_model.fit(X_train, y_train)



# Evaluate KNN
y_pred_knn = knn_model.predict(X_test)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
r2_knn = r2_score(y_test, y_pred_knn)
print(f"KNN -> RMSE: {rmse_knn:.4f}, R²: {r2_knn:.4f}")

# 6. ฝึก Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest -> RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

# 7. บันทึกโมเดลและ scaler
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'feature_columns.pkl')

print("โมเดลและ scaler ถูกบันทึกเรียบร้อยแล้ว!")