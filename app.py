import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

st.title("Diabetes Prediction App")
st.write("Enter the following health measurements to predict the diabetes outcome:")

# Input widgets for each feature
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, value=33)

if st.button("Predict Diabetes Outcome"):
    # Assemble the input data into a NumPy array (shape: (1,8))
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    
    # Load scaler and models
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("logreg_model.pkl", "rb") as f:
            logreg = pickle.load(f)
        nn_model = tf.keras.models.load_model("nn_model.h5")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Get predicted probabilities from both models
    lr_prob = logreg.predict_proba(input_scaled)[0][1]
    nn_prob = nn_model.predict(input_scaled).flatten()[0]
    
    # Ensemble: average the probabilities
    ensemble_prob = (lr_prob + nn_prob) / 2
    # Final prediction based on a 0.5 threshold
    ensemble_pred = "Diabetic" if ensemble_prob >= 0.5 else "Non-diabetic"
    
    st.write("### Final Prediction:")
    st.write(f"The patient is **{ensemble_pred}** (Ensemble Probability: {ensemble_prob:.2f}).")
