import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load the trained model

# Load the trained model
model = joblib.load("fuel_efficiency_model.pkl")

# Title
st.title("🚗 Fuel Efficiency Analysis & Prediction")

# Load Data
df = pd.read_csv("mpg_raw.csv")

# Display dataset
st.subheader("📂 Raw Data")
st.write(df.head())

# Basic statistics
st.subheader("📊 Basic Statistics")
st.write(df.describe())

# Select columns to visualize
selected_columns = st.multiselect("📌 Select columns to display", df.columns)

if selected_columns:
    st.subheader("🔎 Selected Columns Data")
    st.write(df[selected_columns])

# -------------------------------------
# 🔹 User Input for Fuel Efficiency Prediction
# -------------------------------------
st.subheader("⚙️ Enter Vehicle Attributes to Predict MPG")

cylinders = st.number_input("Cylinders", min_value=2, max_value=12, value=4, step=1)
displacement = st.number_input("Displacement", min_value=50.0, max_value=500.0, value=150.0, step=1.0)
horsepower = st.number_input("Horsepower", min_value=50, max_value=500, value=100, step=1)
weight = st.number_input("Weight (lbs)", min_value=1000, max_value=6000, value=2500, step=50)
acceleration = st.number_input("Acceleration", min_value=5.0, max_value=25.0, value=10.0, step=0.1)
model_year = st.number_input("Model Year", min_value=1970, max_value=2025, value=2020, step=1)
origin = st.selectbox("Origin", ["USA", "Europe", "Japan"])

# Convert categorical origin to numeric
origin_map = {"USA": 1, "Europe": 2, "Japan": 3}
origin = origin_map[origin]

# Prediction button
if st.button("🚀 Predict mileage"):
    # Prepare input data for model
    input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
    
    # Make prediction
    predicted_mpg = model.predict(input_data)[0]
    
    # Display result
    st.success(f"🎯 Predicted Fuel Efficiency: {predicted_mpg:.2f} MPG")
