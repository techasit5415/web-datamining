import streamlit as st

st.title("Welcome to Sleep Quality Prediction 😴")
st.header("G1 Team")
st.write("This is a simple web application that predicts your sleep quality based on your daily activities and health data.")

# st.page_link("configpage.py", label="Landing", icon="🛬")
st.page_link("Pages/Frontend.py", label="Form to test", icon="1️⃣")
st.page_link("Pages/model.py", label="Training", icon="2️⃣")
