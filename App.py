import streamlit as st
import pickle
import numpy as np

with open("logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Diabetes Prediction App")
st.write("Logistic Regression Model Deployment")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1, step=1)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f" Diabetic (Probability: {probability:.2f})")
    else:
        st.success(f" Not Diabetic (Probability: {probability:.2f})")
        
