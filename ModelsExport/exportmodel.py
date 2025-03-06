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
import os
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb  # Add XGBoost import

# Load data
file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

# Data cleaning
df['Sleep Disorder'] = df['Sleep Disorder'].replace(np.nan, 'None')
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]
df = df[(df['Physical Activity Level (minutes/day)'] > 0) | (df['Daily Steps'] == 0)]

# Select specific features
selected_columns = [
    'Age',
    'Sleep Duration (hours)',
    'Physical Activity Level (minutes/day)',
    'Stress Level (scale: 1-10)',
    'Heart Rate (bpm)',
    'Daily Steps',
    'Blood Pressure (systolic/diastolic)',
    'Quality of Sleep (scale: 1-10)'  # Target
]

df = df[selected_columns]

# Split blood pressure into systolic and diastolic
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)
df = df.drop(columns=['Blood Pressure (systolic/diastolic)'])

# Final features after preprocessing
features = [
    'Age',
    'Sleep Duration (hours)',
    'Physical Activity Level (minutes/day)',
    'Stress Level (scale: 1-10)',
    'Heart Rate (bpm)',
    'Daily Steps',
    'Systolic BP',
    'Diastolic BP'
]

# Split features and target
X = df[features]
y = df['Quality of Sleep (scale: 1-10)']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature Selection using Random Forest
selector_rf = RandomForestRegressor(n_estimators=200, random_state=42)
selector_rf.fit(X_train, y_train)
importance_rf = selector_rf.feature_importances_

# Select features using Random Forest (modified this part)
rf_selector = SelectFromModel(selector_rf, prefit=True, threshold="mean", max_features=len(features))  # Added prefit=True
X_train_selected = rf_selector.transform(X_train)
X_test_selected = rf_selector.transform(X_test)

# Train models
models = {
    'xgb_model': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    'rf_model': RandomForestRegressor(n_estimators=200, random_state=42),
    'ridge_model': Ridge(alpha=10)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Save models and preprocessing objects
models_dir = r'C:\Users\techa\OneDrive\เอกสาร\VScode\y.2\datamining\web-datamining\Models'
os.makedirs(models_dir, exist_ok=True)

for name, model in models.items():
    joblib.dump(model, os.path.join(models_dir, f'{name}.pkl'))

joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(features, os.path.join(models_dir, 'feature_columns.pkl'))
joblib.dump(rf_selector, os.path.join(models_dir, 'feature_selector.pkl'))

print("✅ Models and preprocessing objects saved successfully!")

# Print feature importances
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': selector_rf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance)
