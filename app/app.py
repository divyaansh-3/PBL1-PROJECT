import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, columns
model = pickle.load(open("../models/model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))
columns = pickle.load(open("../models/columns.pkl", "rb"))

# Page config
st.set_page_config(page_title="ICU Prediction", layout="centered")

# Title
st.title("🏥 ICU Mortality Prediction System")
st.markdown("Predict patient survival risk using Machine Learning")

# Show model info
st.write("Model loaded successfully")

st.divider()

# INPUT SECTION
st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 100)
    heart_rate = st.number_input("Heart Rate")
    blood_pressure = st.number_input("Blood Pressure")
    respiratory_rate = st.number_input("Respiratory Rate")

with col2:
    oxygen_saturation = st.number_input("Oxygen Saturation")
    temperature = st.number_input("Temperature")
    glucose = st.number_input("Glucose Level")

# Threshold slider
threshold = st.slider("Select Risk Threshold", 0.0, 1.0, 0.5)

st.divider()

# PREDICTION
if st.button("Predict", use_container_width=True):

    input_dict = {
        "age": age,
        "heart_rate": heart_rate,
        "blood_pressure": blood_pressure,
        "respiratory_rate": respiratory_rate,
        "oxygen_saturation": oxygen_saturation,
        "temperature": temperature,
        "glucose_level": glucose
    }

    # Create full feature set
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    for col in input_dict:
        if col in input_df.columns:
            input_df[col] = input_dict[col]

    # Default categorical values
    if "gender_Male" in input_df.columns:
        input_df["gender_Male"] = 1
    if "admission_type_Emergency" in input_df.columns:
        input_df["admission_type_Emergency"] = 1

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob >= threshold else 0

    # Risk category
    if prob < 0.3:
        risk = "Low Risk"
        color = "green"
    elif prob < 0.7:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # OUTPUT
    st.subheader("Prediction Result")

    st.markdown(f"### Risk Level: :{color}[{risk}]")
    st.write(f"**Probability of Death:** {prob:.2f}")
    st.write(f"**Predicted Class (0=Alive, 1=Death):** {prediction}")

    # Progress bar
    st.progress(float(prob))

    # Risk message
    if risk == "High Risk":
        st.error("⚠️ Immediate medical attention required")
    elif risk == "Medium Risk":
        st.warning("Monitor patient closely")
    else:
        st.success("Patient condition stable")

# FEATURE IMPORTANCE
st.divider()
st.subheader("Top Contributing Factors")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = columns

    top_features = sorted(zip(importances, feature_names), reverse=True)[:5]

    for imp, name in top_features:
        st.write(f"{name}: {imp:.3f}")
else:
    st.write("Feature importance not available for this model (Logistic Regression).")

# MODEL INFO
st.divider()
st.subheader("Model Info")

st.write("""
- Model Type: Machine Learning (Logistic Regression / Random Forest)
- Input: ICU patient vitals
- Output: Mortality probability
- Purpose: Clinical decision support system
""")

# WARNING
st.warning("⚠️ This system is for decision support only and not a replacement for medical professionals.")