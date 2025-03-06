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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor  # เพิ่ม KNN
from sklearn.linear_model import LinearRegression, Ridge, Lasso ,LassoCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint,uniform
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


# เชื่อมต่อ Google Drive

# ตั้ง path ไปยังไฟล์ที่บันทึกใน Google Drive
file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"

# Add title and description
st.title("Sleep Quality Prediction Models")
st.write("This application analyzes sleep quality data using various machine learning models.")

# Wrap all data processing and model training in a spinner
with st.spinner("Processing data and training models...", show_time=True):
    # Data loading
    file_path = "C:\\Users\\techa\\Downloads\\sleep_health_lifestyle_dataset.csv"
    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

    # Display data overview
    st.header("Data Overview")
    st.write("Dataset Shape:", df.shape)
    st.write("First Few Rows:")
    st.dataframe(df.head())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Data Preprocessing
    # Data cleaning
    # Missing value
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

    # Data Transformation

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
    rf_selector = SelectFromModel(selector_rf, threshold="mean", max_features=10)
    X_train_selected_rf = rf_selector.transform(X_train)
    X_test_selected_rf = rf_selector.transform(X_test)

    # สำหรับ XGBoost
    xgb_selector = SelectFromModel(selector_xgb, threshold="mean", max_features=10)
    X_train_selected_xgb = xgb_selector.transform(X_train)
    X_test_selected_xgb = xgb_selector.transform(X_test)

    # แสดงฟีเจอร์ที่เลือกจากทั้งสองโมเดล
    selected_features_rf = X.columns[rf_selector.get_support()]
    selected_features_xgb = X.columns[xgb_selector.get_support()]

    print("Selected features from Random Forest:", selected_features_rf)
    print("Selected features from XGBoost:", selected_features_xgb)

    # ลองใช้ feature selection จาก Random forest กับ random forest
    # ตั้งค่าพารามิเตอร์สำหรับ Random Forest
    rf_params = {
        'n_estimators': randint(100, 300),
        'max_depth': [None, 10, 20],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf':randint(1, 5)
    }

    # ใช้ GridSearch เพื่อหา params ที่ดีที่สุดของ Random Forest
    rf_random = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params, n_iter=20, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    rf_random.fit(X_train_selected_rf, y_train)

    # Train Model ที่ดีที่สุดของ Random Forest
    rf_model = rf_random.best_estimator_
    y_pred_rf = rf_model.predict(X_test_selected_rf)

    # Evaluate Random Forest
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f" Random Forest -> RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

    # ลองใช้ feature selection จาก xg boost กับ โมเดล random forest

    rf_random.fit(X_train_selected_xgb, y_train)
    # Train Model ที่ดีที่สุดของ Random Forest
    rf_model1 = rf_random.best_estimator_
    y_pred_rf1 = rf_model1.predict(X_test_selected_xgb)

    # Evaluate Random Forest
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf1))
    r2_rf = r2_score(y_test, y_pred_rf1)

    print(f" Random Forest -> RMSE: {rmse_rf:.4f}, R²: {r2_rf:.4f}")

    # ลองใช้ feature selection จาก XGboost กับ โมเดล XGboost
    # ✅ตั้งค่าพารามิเตอร์สำหรับ XGBoost
    xgb_params = {
        'n_estimators': randint(100, 300),      # สุ่มค่า 100 - 300 ต้นไม้
        'max_depth': randint(3, 10),            # ความลึกของต้นไม้ 3 - 10
        'learning_rate': uniform(0.01, 0.3),    # ค่าการเรียนรู้ 0.01 - 0.3
        'subsample': uniform(0.6, 0.4)          # สุ่ม subsample ระหว่าง 0.6 - 1.0
    }

    # ใช้ GridSearch เพื่อหา params ที่ดีที่สุดของ XGBoost
    xgb_random = RandomizedSearchCV(xgb.XGBRegressor(random_state=42),xgb_params,n_iter=20, cv=5,scoring='neg_mean_squared_error',n_jobs=-1,random_state=42)
    xgb_random.fit(X_train_selected_xgb, y_train)

    # Train Model ที่ดีที่สุดของ XGBoost
    xgb_model1 = xgb_random.best_estimator_
    y_pred_xgb1 = xgb_model1.predict(X_test_selected_xgb)

    # Evaluate XGBoost
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb1))
    r2_xgb = r2_score(y_test, y_pred_xgb1)

    print(f"XGBoost -> RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

    # ลองใช้ feature selection จาก Random forest กับ โมเดล XGboost
    xgb_random.fit(X_train_selected_rf, y_train)

    # Train Model ที่ดีที่สุดของ XGBoost
    xgb_model = xgb_random.best_estimator_
    y_pred_xgb = xgb_model.predict(X_test_selected_rf)

    # Evaluate XGBoost
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f"XGBoost -> RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")

    # ✅ เลือก Feature Selection ที่ดีที่สุด
    X_train_final = X_train_selected_rf
    X_test_final = X_test_selected_rf

    print(X_train_final.shape)
    print(X_test_final.shape)

    # Prediction

    # 1. XGBoost
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # 2. LightGBM

    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42,verbose=-1)
    lgb_model.fit(X_train_final, y_train)
    y_pred_lgb = lgb_model.predict(X_test_final)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    r2_lgb = r2_score(y_test, y_pred_lgb)

    # 3. Random forest
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

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    import numpy as np

    # กำหนดค่า alpha
    alpha_value = 10

    # สร้าง Ridge Regression Model
    ridge_model = Ridge(alpha=alpha_value)
    ridge_model.fit(X_train_final, y_train)  # Fit the model first

    # ใช้ Cross-validation (cv=5 folds) วัดค่าความผิดพลาด (RMSE)
    rmse_scores = -cross_val_score(ridge_model, X_train_final, y_train, cv=5, scoring='neg_root_mean_squared_error')

    # แสดงผลลัพธ์
    print(f"Alpha: {alpha_value}")
    print(f"RMSE scores from CV: {rmse_scores}")
    print(f"Mean RMSE: {np.mean(rmse_scores)}")
    print(f"Standard Deviation RMSE: {np.std(rmse_scores)}")

    # Register all models after they're properly fitted
    reg_models = {
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "Random Forest": rf_model,
        "Linear Regression": lr_model,
        "Ridge Regression": ridge_model,
        "KNN Regression": knn_model
    }

    # Ensure all models are fitted before visualization
    for model_name, model in reg_models.items():
        if not hasattr(model, 'coef_'):  # Check if model needs fitting
            model.fit(X_train_final, y_train)

# Now create the UI components
st.header("Data Overview")
st.write("Dataset Shape:", df.shape)
st.write("First Few Rows:")
st.dataframe(df.head())
st.write("Missing Values:")
st.write(df.isnull().sum())

# Create tabs for different sections
st.header("Model Selection")
comparison_mode = st.radio("Comparison Mode", ["Single Model", "Multiple Models"])

if comparison_mode == "Single Model":
    selected_model = st.selectbox("Select Model", results_df['Model'].tolist())
    filtered_results = results_df[results_df['Model'] == selected_model]
else:
    selected_models_list = st.multiselect("Select Models to Compare", 
                                        results_df['Model'].tolist(),
                                        default=results_df['Model'].tolist()[:3])
    filtered_results = results_df[results_df['Model'].isin(selected_models_list)]

tab1, tab2, tab3 = st.tabs(["Model Results", "Feature Importance", "Predictions Visualization"])

with tab1:
    st.header("Model Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Results Table")
        st.dataframe(filtered_results.style.highlight_max(axis=0))
    
    with col2:
        st.subheader("Performance Metrics")
        fig1 = go.Figure(data=[
            go.Bar(name='RMSE', x=filtered_results['Model'], y=filtered_results['RMSE']),
            go.Bar(name='R² Score', x=filtered_results['Model'], y=filtered_results['R² Score'])
        ])
        fig1.update_layout(title='Model Performance Metrics',
                          barmode='group',
                          xaxis_tickangle=-45,
                          height=400)
        st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Feature Importance")
    # Only show feature importance for selected models
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest Features")
        if "Random Forest" in filtered_results['Model'].values:
            st.write(selected_features_rf)
        else:
            st.info("Select Random Forest model to view its features")
    with col2:
        st.subheader("XGBoost Features")
        if "XGBoost" in filtered_results['Model'].values:
            st.write(selected_features_xgb)
        else:
            st.info("Select XGBoost model to view its features")

with tab3:
    st.header("Predictions Visualization")
    
    # Filter models for visualization
    filtered_models = {k: v for k, v in reg_models.items() 
                      if k in filtered_results['Model'].values}
    
    # Create scatter plot for actual vs predicted
    fig2 = go.Figure()
    
    for model_name, model in filtered_models.items():
        y_pred = model.predict(X_test_final)
        fig2.add_trace(
            go.Scatter(x=y_test, y=y_pred,
                      mode='markers',
                      name=f'{model_name}',
                      opacity=0.6)
        )
    
    fig2.add_trace(
        go.Scatter(x=[y_test.min(), y_test.max()],
                  y=[y_test.min(), y_test.max()],
                  mode='lines',
                  name='Ideal',
                  line=dict(color='red', dash='dash'))
    )
    
    fig2.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values'
    )
    
    st.plotly_chart(fig2)

# Add cross-validation results
st.header("Cross-validation Results")
cv_results = pd.DataFrame({
    "Fold": range(1, len(rmse_scores) + 1),
    "RMSE": rmse_scores
})
st.dataframe(cv_results)
st.write(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
st.write(f"Standard Deviation RMSE: {np.std(rmse_scores):.4f}")

# สร้างกราฟเปรียบเทียบค่า Actual vs Predicted สำหรับโมเดล Regression
plt.figure(figsize=(10, 7))
for model_name, model in reg_models.items():
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)  # ใช้ X_test แทน

    # Plot จุดการทำนายและค่าจริง
    plt.scatter(y_test, y_pred, alpha=0.6, label=f"{model_name} Predictions")

# Plot เส้นอุดมคติ (Ideal Line)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal")

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression: Actual vs Predicted (All Models)")
plt.legend()
plt.grid()
plt.show()

# ตั้งค่าขนาดของกราฟ
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 🔹 Bar Plot ของ RMSE
sns.barplot(data=results_df, x="Model", y="RMSE", ax=axes[0], palette="Blues_r")
axes[0].set_title("Comparison of RMSE for Regression Models")
axes[0].set_ylabel("RMSE")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")

# 🔹 Bar Plot ของ R² Score
sns.barplot(data=results_df, x="Model", y="R² Score", ax=axes[1], palette="Greens_r")
axes[1].set_title("Comparison of R² Score for Regression Models")
axes[1].set_ylabel("R² Score")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.show()

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

st.header("Additional Comparative Analysis")

# 1. Model Performance Distribution
fig_box_perf = go.Figure()
fig_box_perf.add_trace(go.Box(y=results_df['RMSE'], name='RMSE'))
fig_box_perf.add_trace(go.Box(y=results_df['R² Score'], name='R² Score'))
fig_box_perf.update_layout(title='Distribution of Model Performance Metrics')
st.plotly_chart(fig_box_perf)

# 2. Feature Importances Comparison
if any(model in filtered_results['Model'].values for model in ['Random Forest', 'XGBoost']):
    st.subheader("Feature Importance Comparison")
    
    importance_data = {}
    feature_names = []
    
    if 'Random Forest' in filtered_results['Model'].values:
        importance_data['Random Forest'] = selector_rf.feature_importances_
        feature_names = X.columns
    
    if 'XGBoost' in filtered_results['Model'].values:
        importance_data['XGBoost'] = selector_xgb.feature_importances_
        feature_names = X.columns
    
    fig_importance = go.Figure()
    for model_name, importances in importance_data.items():
        fig_importance.add_trace(go.Bar(
            name=model_name,
            x=feature_names,
            y=importances
        ))
    
    fig_importance.update_layout(
        title='Feature Importance by Model',
        xaxis_tickangle=-45,
        barmode='group'
    )
    st.plotly_chart(fig_importance)

# 3. Prediction Distribution
predictions_dict = {}
for model_name, model in filtered_models.items():
    predictions_dict[model_name] = model.predict(X_test_final)

predictions_df = pd.DataFrame(predictions_dict)
predictions_df['Actual'] = y_test.values

fig_dist = go.Figure()
for col in predictions_df.columns:
    fig_dist.add_trace(go.Violin(
        y=predictions_df[col],
        name=col,
        box_visible=True,
        meanline_visible=True
    ))

fig_dist.update_layout(title='Distribution of Predictions vs Actual Values')
st.plotly_chart(fig_dist)

# 4. Error Analysis
st.subheader("Error Analysis")
error_df = pd.DataFrame()
for model_name in filtered_models.keys():
    error_df[f'{model_name} Error'] = predictions_df[model_name] - predictions_df['Actual']

fig_error = go.Figure()
for col in error_df.columns:
    fig_error.add_trace(go.Box(y=error_df[col], name=col))

fig_error.update_layout(title='Prediction Error Distribution by Model')
st.plotly_chart(fig_error)

# 5. Prediction Scatter Matrix
if len(filtered_models) > 1:
    st.subheader("Prediction Correlation Matrix")
    fig_scatter = px.scatter_matrix(predictions_df)
    fig_scatter.update_layout(title='Scatter Matrix of Predictions')
    st.plotly_chart(fig_scatter)

# 6. Time Series-like Comparison
st.subheader("Sequential Prediction Comparison")
fig_seq = go.Figure()

# Get predictions and sort them along with actual values
predictions_and_actual = pd.DataFrame({'Actual': y_test})
for model_name, model in filtered_models.items():
    y_pred = model.predict(X_test_final)
    predictions_and_actual[model_name] = y_pred

# Sort by actual values
predictions_and_actual = predictions_and_actual.sort_values('Actual')

# Add traces for each model
for model_name in filtered_models.keys():
    fig_seq.add_trace(go.Scatter(
        y=predictions_and_actual[model_name],
        name=model_name,
        mode='lines'
    ))

# Add actual values
fig_seq.add_trace(go.Scatter(
    y=predictions_and_actual['Actual'],
    name='Actual',
    mode='lines',
    line=dict(color='black', dash='dash')
))

fig_seq.update_layout(
    title='Sequential Prediction Comparison',
    xaxis_title='Sorted Sample Index',
    yaxis_title='Sleep Quality'
)
st.plotly_chart(fig_seq)

# 7. Regression Metrics Comparison
st.subheader("Detailed Metrics Comparison")

# Calculate metrics only for filtered models
metrics_list = []
for model_name in filtered_results['Model']:
    if model_name in reg_models:
        model = reg_models[model_name]
        y_pred = model.predict(X_test_final)
        metrics_list.append({
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),  # Changed this line
            'R² Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred)
        })

metrics_comparison = pd.DataFrame(metrics_list)

if not metrics_comparison.empty:
    # Create parallel coordinates plot with numeric color scale
    fig_metrics = px.parallel_coordinates(
        metrics_comparison,
        dimensions=['RMSE', 'R² Score', 'MAE'],
        color=np.arange(len(metrics_comparison)),
        color_continuous_scale='Viridis'
    )

    fig_metrics.update_layout(
        title='Parallel Coordinates Plot of Model Metrics',
        coloraxis_showscale=False
    )

    # Add model names as annotations
    for i, model in enumerate(metrics_comparison['Model']):
        fig_metrics.add_annotation(
            x=-0.1,
            y=i,
            text=model,
            showarrow=False,
            xanchor='right'
        )

    st.plotly_chart(fig_metrics)
else:
    st.info("Please select models to compare their metrics")