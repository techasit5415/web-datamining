import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score,r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, Ridge, Lasso

st.title("Sleep Health and Lifestyle")


file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"

# โหลดไฟล์จาก Drive
df = pd.read_csv(file_path, encoding='latin1', low_memory=False) #
st.write(df)
st.write(df.isnull().sum(),df.info())

#------------------------------------------------------------------------------------------------------

#data cleaning
df['Sleep Disorder'].unique()

#------------------------------------------------------------------------------------------------------

df['Sleep Disorder'].replace(np.nan, 'None', inplace=True)

#------------------------------------------------------------------------------------------------------

df['Sleep Disorder'].unique()

#------------------------------------------------------------------------------------------------------

df.isnull().sum()

#------------------------------------------------------------------------------------------------------

# ลบค่าที่อยู่นอกช่วงปกติของข้อมูล เช่น อายุ < 0 หรือ > 120
df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

# ตรวจสอบความสมเหตุสมผลของ Daily Steps และ Physical Activity Level
df = df[(df['Physical Activity Level (minutes/day)'] > 0) | (df['Daily Steps'] == 0)]

#------------------------------------------------------------------------------------------------------

df = df.drop(columns=['Person ID'])

# One-Hot Encoding สำหรับตัวแปรประเภท (Gender, Occupation, BMI Category, Sleep Disorder)
df_encoded = pd.get_dummies(df, columns=['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'], drop_first=True)

# แยกค่าความดันโลหิต (Blood Pressure) ออกเป็น Systolic BP และ Diastolic BP
df_encoded[['Systolic BP', 'Diastolic BP']] = df_encoded['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)

# ลบคอลัมน์เดิมที่เป็นข้อความ
df_encoded = df_encoded.drop(columns=['Blood Pressure (systolic/diastolic)'])

# แสดงผลลัพธ์
st.write((df_encoded.info()),(df_encoded.head()))
print(df_encoded.info())
print(df_encoded.head())

#Data reduction (Feature Selection)


#------------------------------------------------------------------------------------------------------
# แยก X และ y
X = df_encoded.drop(columns=['Quality of Sleep (scale: 1-10)'])
y = df_encoded['Quality of Sleep (scale: 1-10)']

# ใช้ RandomForest เพื่อเลือกฟีเจอร์
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# เลือก 8 ฟีเจอร์ที่สำคัญที่สุด
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
important_features_rf = feature_importance.nlargest(10).index

print(f"Selected Features (RandomForest): {list(important_features_rf)}")

# ใช้เฉพาะฟีเจอร์ที่เลือก
X_selected_rf = X[important_features_rf]

#------------------------------------------------------------------------------------------------------

# ใช้ XGBoost เพื่อเลือกฟีเจอร์
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X, y)

# เลือก 10 ฟีเจอร์ที่สำคัญที่สุด
feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
important_features_xgb = feature_importance.nlargest(10).index

print(f"Selected Features (XGBoost): {list(important_features_xgb)}")

# ใช้เฉพาะฟีเจอร์ที่เลือก
X_selected_xgb = X[important_features_xgb]

#------------------------------------------------------------------------------------------------------

# แบ่งข้อมูล 80% เทรน, 20% เทส
X_train, X_test, y_train, y_test = train_test_split(X_selected_xgb, y, test_size=0.2, random_state=42)

# Standardization (ปรับข้อมูลให้อยู่ในช่วงเดียวกัน)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#------------------------------------------------------------------------------------------------------

# ตั้งค่าพารามิเตอร์สำหรับปรับแต่ง
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ใช้ GridSearch เพื่อหา params ที่ดีที่สุด
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

# Train Model ที่ดีที่สุด
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test_scaled)

# Evaluate
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"RandomForest RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

#------------------------------------------------------------------------------------------------------

# ตั้งค่าพารามิเตอร์สำหรับ XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0]
}

# ใช้ GridSearch
xgb_grid = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)

# Train Model ที่ดีที่สุด
xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test_scaled)

# Evaluate
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

#------------------------------------------------------------------------------------------------------

#Data transformation
#ปรับเสกล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_selected_xgb)

#------------------------------------------------------------------------------------------------------

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=0.2, random_state=42)

# 1. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# 2. LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)


# 3. RandomForest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)



# 5. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# 6. Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)


# 7. Lasso Regression
from sklearn.linear_model import LassoCV
# ค้นหา alpha ที่เหมาะสมโดยใช้ Cross-Validation
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

# ใช้ alpha ที่ดีที่สุด
best_alpha = lasso_cv.alpha_

lasso_model = Lasso(alpha=best_alpha)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print(f"Best alpha: {best_alpha}")
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)




# แสดงผล RMSE ของแต่ละโมเดล
# เก็บผลลัพธ์ลง DataFrame
results_df = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "Random Forest", "Linear Regression", "Ridge Regression", "Lasso Regression"],
    "RMSE": [rmse_xgb, rmse_lgb, rmse_rf, rmse_lr, rmse_ridge, rmse_lasso],
    "R² Score": [r2_xgb, r2_lgb, r2_rf, r2_lr, r2_ridge, r2_lasso]
})

# แสดงผลลัพธ์เป็นข้อความ
print(results_df)
st.write(f"Best alpha for Lasso: {best_alpha}")

#------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

reg_models = {
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "Random Forest": rf_model,
    "Linear Regression": lr_model,
    "Ridge Regression": ridge_model,
    "Lasso Regression": lasso_model
}
# สร้างกราฟเปรียบเทียบค่า Actual vs Predicted สำหรับโมเดล Regression
plt.figure(figsize=(10, 7))
for model_name, model in reg_models.items():
    model.fit(X_train, y_train)  # ใช้ X_train และ y_train แทน
    y_pred = model.predict(X_test)  # ใช้ X_test แทน

    # Plot จุดการทำนายและค่าจริง
    plt.scatter(y_test, y_pred, alpha=0.6, label=f"{model_name} Predictions")

# Plot เส้นอุดมคติ (Ideal Line)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal")

plt.xlabel("Actual Values")
# st.write(plt.xlabel("Actual Values"))
plt.ylabel("Predicted Values")
# st.write(plt.ylabel("Predicted Values"))
plt.title("Regression: Actual vs Predicted (All Models)")
# st.write(plt.title("Regression: Actual vs Predicted (All Models)"))
plt.legend()
# st.write(plt.legend())
plt.grid()
# st.write(plt.grid())
fig = plt.show()
st.pyplot(fig)
#------------------------------------------------------------------------------------------------------

option2 = st.selectbox("Gender ",("Male", "Female"))
Feture2=st.number_input("Age",min_value=1,max_value=100)
option3 = st.selectbox("Occupation",("Office Worker", "Student","Manual Labor", "Retired"))
Feture3=st.number_input("Sleep Duration(Hour)",min_value=0.0,max_value=24.0)
Feture4=st.number_input("Quality Of Sleep(Scale 1-10)",min_value=1,max_value=10)
Feture5=st.number_input("Phsical Activity Level (Min/Day)",min_value=1,max_value=300)
Feture5=st.number_input("Stress Level (Scale 1-10)",min_value=1,max_value=10)
# Feture6=st.number_input("BMI Category",min_value=1,max_value=10)
option1 = st.selectbox("BMI Category",("Underweight", "Normal","Obese", "Overweight"))
Feture7=st.text_input("Blood Pressure (systolic/diastolic) ")
Feture8=st.number_input("Heart Rate (bpm)",min_value=1,max_value=300)
Feture9=st.number_input("Daily Steps,",min_value=0)
option = st.selectbox("Sleep Disorder",("None", "Insomnia", "Sleep Apnea"))
# Feture10=st.number_input("Sleep Disorder",min_value=1,max_value=10)


# dataframe = np.random.randn(10, 3)
# chart_data = pd.DataFrame(dataframe,columns=['a', 'b', 'c'])

# st.line_chart(chart_data)
# st.dataframe(dataframe)
# st.snow()

