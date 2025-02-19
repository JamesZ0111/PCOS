import streamlit as st
from joblib import load
import joblib
import json
import pandas as pd
import numpy as np

# Load model, scaler, imputer, and feature columns
model = joblib.load('model_new.joblib')
scaler = joblib.load('scaler_new.joblib')
imputer = joblib.load('imputer_new.joblib')

with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# App layout
st.set_page_config(page_title="PCOS Prediction System", layout="wide")
st.title("PCOS Prediction System")

# Navigation Bar
st.sidebar.title("Navigation")
view_option = st.sidebar.radio("Choose Version", ["Basic", "Premium"])

if view_option == "Basic":
    st.header("Basic PCOS Prediction")
    
    with st.form(key='pcos_prediction_form'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)")
            weight = st.number_input("Weight (Kg)")
            height = st.number_input("Height (Cm)")
            bmi = st.number_input("BMI", min_value=15.0, max_value=45.0)
            pulse_rate = st.number_input("Pulse rate (bpm)", min_value=50, max_value=150)
            cycle = st.selectbox("Cycle", options=[1, 2], format_func=lambda x: "Regular" if x == 1 else "Irregular")
            cycle_length = st.number_input("Period length (days)", min_value=1, max_value=20)
            marriage_status = st.number_input("Marriage Status (Years)", min_value=0, max_value=50)
            pregnant = st.selectbox("Pregnant", options=['N', 'Y'])
        
        with col2:
            abortions = st.number_input("Number of Abortions", min_value=0, max_value=10)
            hip = st.number_input("Hip (inch)", min_value=30, max_value=60)
            waist = st.number_input("Waist (inch)", min_value=20, max_value=60)
            waist_hip_ratio = st.number_input("Waist:Hip Ratio", min_value=0.5, max_value=1.5)
            weight_gain = st.selectbox("Weight Gain", options=['N', 'Y'])
            hair_growth = st.selectbox("Hair Growth", options=['N', 'Y'])
            skin_darkening = st.selectbox("Skin Darkening", options=['N', 'Y'])
            hair_loss = st.selectbox("Hair Loss", options=['N', 'Y'])
            pimples = st.selectbox("Pimples", options=['N', 'Y'])
            fast_food = st.selectbox("Fast Food", options=['N', 'Y'])
            regular_exercise = st.selectbox("Regular Exercise", options=['N', 'Y'])

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        input_data = pd.DataFrame([{
            'Age (yrs)': age, 'Weight (Kg)': weight, 'Height(Cm)': height, 'BMI': bmi,
            'Pulse rate(bpm)': pulse_rate, 'Cycle(R/I)': cycle, 'Cycle length(days)': cycle_length,
            'Marraige Status (Yrs)': marriage_status, 'Pregnant(Y/N)': pregnant, 'No. of aborptions': abortions,
            'Hip(inch)': hip, 'Waist(inch)': waist, 'Waist:Hip Ratio': waist_hip_ratio,
            'Weight gain(Y/N)': weight_gain, 'hair growth(Y/N)': hair_growth,
            'Skin darkening (Y/N)': skin_darkening, 'Hair loss(Y/N)': hair_loss,
            'Pimples(Y/N)': pimples, 'Fast food (Y/N)': fast_food, 'Reg.Exercise(Y/N)': regular_exercise
        }])

        categorical_columns = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
                               'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)']
        special_columns = ['Cycle(R/I)']

        for col in categorical_columns:
            input_data[col] = input_data[col].map({'Y': 1, 'N': 0})
        for col in special_columns:
            input_data[col] = input_data[col].map({1: 2, 2: 4})

        input_data = input_data[feature_columns]
        input_data = pd.DataFrame(imputer.transform(input_data), columns=feature_columns)
        input_data_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)
        
        st.subheader("Prediction Result")
        st.write("**Prediction:**", "Likely to have PCOS" if prediction[0] == 1 else "Unlikely to have PCOS")
        st.write("**Probability of PCOS:**", f"{prediction_proba[0][1]:.2%}")

elif view_option == "Premium":
    st.header("Premium PCOS Prediction")
    st.subheader("Advanced AI Model Coming Soon!")
    st.info("We are currently working on improving our prediction system with a more advanced AI model.")
