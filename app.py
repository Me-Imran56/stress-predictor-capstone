import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Daily Stress Predictor", page_icon="🧘", layout="centered")

@st.cache_resource
def load_pipeline():
    return joblib.load("stress_pipeline.joblib")

pipeline = load_pipeline()

st.title("🧘 Daily Stress Level Predictor")
st.write(
    "Estimate a 1–10 stress score from everyday lifestyle habits — sleep, activity, and "
    "vitals most fitness trackers already collect. Built for the Neurofive ML Track capstone."
)
st.caption(
    "⚠️ Trained on a synthetic dataset for learning purposes — treat predictions as "
    "illustrative, not a clinical assessment. See the notebook for the full methodology "
    "and an honest discussion of this limitation."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", [
        "Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
        "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"
    ])
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])

with col2:
    sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, step=0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    activity_level = st.slider("Physical Activity Level (minutes/day)", 0, 100, 45)
    heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 100, 70)
    daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=20000, value=6000, step=500)

st.markdown("**Blood Pressure**")
bp_col1, bp_col2 = st.columns(2)
with bp_col1:
    systolic = st.number_input("Systolic", min_value=90, max_value=180, value=120)
with bp_col2:
    diastolic = st.number_input("Diastolic", min_value=60, max_value=120, value=80)

st.divider()

if st.button("Predict Stress Level", type="primary", use_container_width=True):
    sleep_efficiency = quality_of_sleep / sleep_duration
    activity_per_1000_steps = activity_level / (daily_steps / 1000)
    pulse_pressure = systolic - diastolic

    input_df = pd.DataFrame([{
        "Age": age,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": activity_level,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Sleep_Efficiency": sleep_efficiency,
        "Activity_per_1000_Steps": activity_per_1000_steps,
        "Pulse_Pressure": pulse_pressure,
        "Gender": gender,
        "Occupation": occupation,
        "BMI Category": bmi_category,
        "Sleep Disorder": sleep_disorder,
    }])

    prediction = pipeline.predict(input_df)[0]
    prediction_clamped = max(1, min(10, prediction))

    if prediction_clamped <= 4:
        st.success(f"### 🟢 Estimated Stress Level: {prediction_clamped:.1f} / 10 — Low")
    elif prediction_clamped <= 6:
        st.warning(f"### 🟡 Estimated Stress Level: {prediction_clamped:.1f} / 10 — Moderate")
    else:
        st.error(f"### 🔴 Estimated Stress Level: {prediction_clamped:.1f} / 10 — Elevated")

    st.caption(
        f"Sleep efficiency (quality/duration): {sleep_efficiency:.2f} | "
        f"Pulse pressure: {pulse_pressure} mmHg"
    )

st.divider()
st.caption(
    "Model: XGBoost Regressor inside a scikit-learn Pipeline (StandardScaler + OneHotEncoder). "
    "Dataset: Sleep Health and Lifestyle Dataset (374 records, synthetic)."
)
