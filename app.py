import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Stroke Prediction App")
st.markdown("Provide the following details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (in mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['0', '1'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.number_input("Max Heart Rate ", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])


if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex_'+ sex: 1,
        'MaxHR': max_hr,
        'ChestPainType_'+ chest_pain: 1,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG_'+ resting_ecg: 1,
        'ExerciseAngina_'+ exercise_angina: 1,
        'Oldpeak': oldpeak,
        'ST_slope_'+ st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("The model predicts that the patient is likely to have a heart stroke.")
    else:
        st.success("The model predicts that the patient is unlikely to have a heart stroke.")