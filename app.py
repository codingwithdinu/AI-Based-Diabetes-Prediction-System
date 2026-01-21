import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction System")

preg = st.number_input("Pregnancies")
glu = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
ins = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("Person is Non-Diabetic")
    else:
        st.error("Person is Diabetic")
