# -*- coding: utf-8 -*-


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score,r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression,SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor  # เพิ่ม KNN
from sklearn.linear_model import LinearRegression, Ridge, Lasso ,LassoCV
from sklearn.linear_model import Ridge
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ตั้ง path ไปยังไฟล์ที่บันทึกใน Google Drive
with st.spinner("Wait for it...", show_time=True):
    file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"

    # โหลดไฟล์จาก Drive
    df = pd.read_csv(file_path, encoding='latin1', low_memory=False) #

    df.head()

    df.isnull().sum()

    df.info()

    # Display data info using Streamlit
    st.title("Sleep Quality Model Training")

    st.header("Data Overview")
    st.write("Dataset Shape:", df.shape)
    st.write("Column Info:")
    st.write(df.info())
    st.write("First Few Rows:")
    st.dataframe(df.head())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    """Data Preprocessing

    """

    #data cleaning
    #Missing value
    df['Sleep Disorder'].unique()

    df['Sleep Disorder'].replace(np.nan, 'None', inplace=True)

    df['Sleep Disorder'].unique()

    df.isnull().sum()

    # Handle Noisy Data

    # ลบค่าที่อยู่นอกช่วงปกติของข้อมูล เช่น อายุ < 0 หรือ > 120
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

    # ตรวจสอบความสมเหตุสมผลของ Daily Steps และ Physical Activity Level
    df = df[(df['Physical Activity Level (minutes/day)'] > 0) | (df['Daily Steps'] == 0)]

    # ลบคอลัมน์ที่ไม่เกี่ยวข้อง เช่น 'Person ID' (ถ้ามี)
    df = df.drop(columns=['Person ID'])

    #Data Transformation


    # One-Hot Encoding สำหรับตัวแปรประเภท (Gender, Occupation, BMI Category, Sleep Disorder)
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'], drop_first=True)

    # แยกค่าความดันโลหิต (Blood Pressure) ออกเป็น Systolic BP และ Diastolic BP
    df_encoded[['Systolic BP', 'Diastolic BP']] = df_encoded['Blood Pressure (systolic/diastolic)'].str.split('/', expand=True).astype(float)

    # ลบคอลัมน์เดิมที่เป็นข้อความ
    df_encoded = df_encoded.drop(columns=['Blood Pressure (systolic/diastolic)'])

    # แยก Features และ Target
    X = df_encoded.drop(columns=['Quality of Sleep (scale: 1-10)'])
    y = df_encoded['Quality of Sleep (scale: 1-10)']

    # ทำ Scaling **ก่อน** Feature Selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # แสดงผลลัพธ์
    print(df_encoded.info())
    print(df_encoded.head())

    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Feature Selection - Random Forest
    selector_rf = RandomForestRegressor(n_estimators=200, random_state=42)
    selector_rf.fit(X_train, y_train)
    importance_rf = selector_rf.feature_importances_

    # Feature Selection - XGBoost
    selector_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    selector_xgb.fit(X_train, y_train)
    importance_xgb = selector_xgb.feature_importances_

    # 2. ใช้ SelectFromModel เพื่อเลือกฟีเจอร์ที่สำคัญจากแต่ละโมเดล

    # สำหรับ Random Forest
    rf_selector = SelectFromModel(selector_rf, threshold="mean", max_features=10)  # เลือกฟีเจอร์ที่สำคัญที่สุด 10 ตัว
    X_train_selected_rf = rf_selector.transform(X_train)
    X_test_selected_rf = rf_selector.transform(X_test)

    # สำหรับ XGBoost
    xgb_selector = SelectFromModel(selector_xgb, threshold="mean", max_features=10)  # เลือกฟีเจอร์ที่สำคัญที่สุด 10 ตัว
    X_train_selected_xgb = xgb_selector.transform(X_train)
    X_test_selected_xgb = xgb_selector.transform(X_test)

    # แสดงฟีเจอร์ที่เลือกจากทั้งสองโมเดล
    selected_features_rf = X.columns[rf_selector.get_support()]
    selected_features_xgb = X.columns[xgb_selector.get_support()]

    print("Selected features from Random Forest:", selected_features_rf)
    print("Selected features from XGBoost:", selected_features_xgb)

    # Display selected features
    st.header("Feature Selection Results")
    st.subheader("Selected Features from Random Forest:")
    st.write(selected_features_rf)
    st.subheader("Selected Features from XGBoost:")
    st.write(selected_features_xgb)

    #  เปรียบเทียบ Feature Selection ระหว่าง Random Forest และ XGBoost


    # ตั้งค่าพารามิเตอร์สำหรับ Random Forest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    #  ใช้ GridSearch เพื่อหา params ที่ดีที่สุดของ Random Forest
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train_selected_rf, y_train)

    #  Train Model ที่ดีที่สุดของ Random Forest
    rf_best = rf_grid.best_estimator_
    y_pred_rf = rf_best.predict(X_test_selected_rf)

    # Evaluate Random Forest
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f" Random Forest -> RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

    # ✅ตั้งค่าพารามิเตอร์สำหรับ XGBoost
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0]
    }

    #  ใช้ GridSearch เพื่อหา params ที่ดีที่สุดของ XGBoost
    xgb_grid = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_grid.fit(X_train_selected_xgb, y_train)

    #  Train Model ที่ดีที่สุดของ XGBoost
    xgb_best = xgb_grid.best_estimator_
    y_pred_xgb = xgb_best.predict(X_test_selected_xgb)

    #  Evaluate XGBoost
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f"XGBoost -> RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

    # Display model evaluation results
    st.header("Model Evaluation")

    # Random Forest Results
    st.subheader("Random Forest Results")
    st.write(f"Random Forest -> RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")
    st.write("Best Parameters:", rf_grid.best_params_)

    # XGBoost Results
    st.subheader("XGBoost Results")
    st.write(f"XGBoost -> RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
    st.write("Best Parameters:", xgb_grid.best_params_)

    # ✅ เลือก Feature Selection ที่ดีที่สุด
    X_train_final = X_train_selected_xgb
    X_test_final = X_test_selected_xgb
    print(X_train_final.shape)
    print(X_test_final.shape)

    #  เลือก Feature Selection ที่ดีที่สุด
    if rmse_xgb < rmse_rf:
        print("\n**เลือก Feature Selection จาก XGBoost**")
        X_train_final = X_train_selected_xgb
        X_test_final = X_test_selected_xgb
    else:
        print("\n **เลือก Feature Selection จาก Random Forest**")
        X_train_final = X_train_selected_rf
        X_test_final = X_test_selected_rf

    """RMSE ของ XGBoost ต่ำกว่าเลยเลือกใช้ XGBoost ในการทำ feature selection

    **Prediction**
    """

    # 1. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train_final, y_train)
    y_pred_xgb = xgb_model.predict(X_test_final)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # 2. LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42,verbose=-1)
    lgb_model.fit(X_train_final, y_train)
    y_pred_lgb = lgb_model.predict(X_test_final)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)


    # 3. RandomForest
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train_final, y_train)
    y_pred_rf = rf_model.predict(X_test_final)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    # 4. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_final, y_train)
    y_pred_lr = lr_model.predict(X_test_final)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    # 5. Ridge Regression
    ridge_model = Ridge(alpha=10)
    ridge_model.fit(X_train_final, y_train)
    y_pred_ridge = ridge_model.predict(X_test_final)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)


    # 7. KNN Regression (เพิ่มเข้ามา)
    n_neighbors_value = 20
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors_value)  # คุณสามารถปรับค่า n_neighbors ได้
    knn_model.fit(X_train_final, y_train)
    y_pred_knn = knn_model.predict(X_test_final)
    rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    r2_knn = r2_score(y_test, y_pred_knn)

    # แสดงผล RMSE ของแต่ละโมเดล
    results_df = pd.DataFrame({
        
        "Model": ["XGBoost", "LightGBM", "Random Forest", "Linear Regression", "Ridge Regression", f"KNN Regression (n_neighbors={n_neighbors_value})"],
        "RMSE": [rmse_xgb, rmse_lgb, rmse_rf, rmse_lr, rmse_ridge, rmse_knn],
        "R² Score": [r2_xgb, r2_lgb, r2_rf, r2_lr, r2_ridge, r2_knn]
    })

    print(results_df)

    # Display final model comparison results
    st.header("Final Model Comparison")
    st.dataframe(results_df)

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    import numpy as np

    # กำหนดค่า alpha
    alpha_value = 10

    # สร้าง Ridge Regression Model
    ridge_model = Ridge(alpha=alpha_value)

    # ใช้ Cross-validation (cv=5 folds) วัดค่าความผิดพลาด (RMSE)
    rmse_scores = -cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

    # แสดงผลลัพธ์
    print(f"Alpha: {alpha_value}")
    print(f"RMSE scores from CV: {rmse_scores}")
    print(f"Mean RMSE: {np.mean(rmse_scores)}")
    print(f"Standard Deviation RMSE: {np.std(rmse_scores)}")

    # Cross-validation results
    st.header("Cross-validation Results (Ridge Regression)")
    cv_results = pd.DataFrame({
        "Fold": range(1, len(rmse_scores) + 1),
        "RMSE": rmse_scores
    })
    st.dataframe(cv_results)
    st.write(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    st.write(f"Standard Deviation RMSE: {np.std(rmse_scores):.4f}")

    # Initialize and fit all models before visualization
    reg_models = {
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "Random Forest": rf_model,
        "Linear Regression": lr_model,
        "Ridge Regression": ridge_model
    }

    # Ensure all models are fitted
    for model_name, model in reg_models.items():
        if not hasattr(model, 'coef_'):  # Check if model needs fitting
            model.fit(X_train_final, y_train)

    # Add model selection UI
    st.header("Model Selection")
    comparison_mode = st.radio(
        "Choose visualization mode:",
        ["Single Model", "Compare Models"]
    )

    if comparison_mode == "Single Model":
        selected_model = st.selectbox(
            "Select a model to visualize:",
            list(reg_models.keys())
        )
        selected_models = {selected_model: reg_models[selected_model]}
    else:
        selected_models_list = st.multiselect(
            "Select models to compare:",
            list(reg_models.keys()),
        default=list(reg_models.keys())[:2]
        )
        selected_models = {model: reg_models[model] for model in selected_models_list}

    # Modify visualization tabs to use selected models
    tab1, tab2, tab3 = st.tabs(["Model Metrics", "Actual vs Predicted", "Predictions Line Plot"])

    with tab1:
        if comparison_mode == "Single Model":
            # Show only selected model's metrics
            filtered_results = results_df[results_df['Model'] == selected_model]
        else:
            # Show selected models' metrics
            filtered_results = results_df[results_df['Model'].isin(selected_models_list)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rmse = px.bar(
                filtered_results,
                x="Model",
                y="RMSE",
                title="RMSE Comparison",
                color="RMSE"
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        with col2:
            fig_r2 = px.bar(
                filtered_results,
                x="Model",
                y="R² Score",
                title="R² Score Comparison",
                color="R² Score"
            )
            st.plotly_chart(fig_r2, use_container_width=True)

    with tab2:
        fig_scatter = go.Figure()
        
        for model_name, model in selected_models.items():
            y_pred = model.predict(X_test_final)
            fig_scatter.add_trace(
                go.Scatter(x=y_test, y=y_pred, mode='markers', 
                        name=f"{model_name}", opacity=0.6)
            )
        
        fig_scatter.add_trace(
            go.Scatter(x=[y_test.min(), y_test.max()], 
                    y=[y_test.min(), y_test.max()],
                    mode='lines', name='Ideal', 
                    line=dict(color='red', dash='dash'))
        )
        
        fig_scatter.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        sorted_indices = y_test.argsort()
        y_test_sorted = y_test.iloc[sorted_indices]
        X_test_final_sorted = X_test_final[sorted_indices]
        
        fig_line = go.Figure()
        
        fig_line.add_trace(
            go.Scatter(x=list(range(len(y_test_sorted))), 
                    y=y_test_sorted,
                    mode='lines', name='Actual Values', 
                    line=dict(color='black', dash='dash'))
        )
        
        for model_name, model in selected_models.items():
            y_pred = model.predict(X_test_final_sorted)
            fig_line.add_trace(
                go.Scatter(x=list(range(len(y_pred))), 
                        y=y_pred,
                        mode='lines+markers', 
                        name=f"{model_name}", 
                        marker=dict(size=3))
            )
        
        fig_line.update_layout(
            title="Predictions Comparison",
            xaxis_title="Data Points (Sorted)",
            yaxis_title="Values"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # สร้างกราฟเส้นเปรียบเทียบค่า Actual vs Predicted สำหรับโมเดล Regression
    plt.figure(figsize=(12, 8))

    # เรียงข้อมูลตาม y_test เพื่อให้กราฟเส้นดูดีขึ้น
    sorted_indices = y_test.argsort()
    y_test_sorted = y_test.iloc[sorted_indices]
    X_test_final_sorted = X_test_final[sorted_indices]  # Use integer indexing here

    for model_name, model in reg_models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final_sorted)

        # Plot เส้นแสดงค่าทำนาย
        plt.plot(y_test_sorted.index, y_pred, label=f"{model_name} Predictions", linestyle='-', marker='o', markersize=3)

    # Plot เส้นแสดงค่าจริง
    plt.plot(y_test_sorted.index, y_test_sorted, label="Actual Values", linestyle='--', color='black')

    plt.xlabel("Data Points (Sorted by Actual Values)")
    plt.ylabel("Values")
    plt.title("Regression: Actual vs Predicted (Line Plot)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()