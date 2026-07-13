import streamlit as st
import pandas as pd
import joblib

# ===============================================
# PAGE CONFIG
# ===============================================

st.set_page_config(
    page_title="COPD AI Prediction",
    page_icon="🫁",
    layout="wide"
)

# ===============================================
# LOAD MODEL
# ===============================================

try:
    model = joblib.load("model.pkl")
except:
    st.error("model.pkl not found!")
    st.stop()

# ===============================================
# TITLE
# ===============================================

st.title("🫁 COPD Risk Prediction System")

st.write(
    """
Predict whether a patient is at **Low** or **High**
risk of COPD using Machine Learning.
"""
)

# ===============================================
# HELPER FUNCTIONS
# ===============================================

def yes_no(label):
    value = st.selectbox(label, ["No", "Yes"])
    return 1 if value == "Yes" else 0


def gender_input():
    value = st.selectbox("Gender", ["Female", "Male"])
    return 1 if value == "Male" else 0


def level_input(label):
    value = st.selectbox(
        label,
        ["Low", "Medium", "High"]
    )

    mapping = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }

    return mapping[value]


def quality_input(label):

    value = st.selectbox(
        label,
        ["Poor", "Average", "Good"]
    )

    mapping = {
        "Poor": 1,
        "Average": 2,
        "Good": 3
    }

    return mapping[value]


# ===============================================
# INPUT FORM
# ===============================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        18,
        100,
        45
    )

    gender = gender_input()

    education_years = st.slider(
        "Education Years",
        0,
        25,
        10
    )

    income_level = level_input("Income Level")

    smoker = yes_no("Smoker")

    smoking_years = st.slider(
        "Smoking Years",
        0,
        60,
        10
    )

    cigarettes_per_day = st.slider(
        "Cigarettes Per Day",
        0,
        60,
        10
    )

    pack_years = st.slider(
        "Pack Years",
        0,
        60,
        5
    )

    passive_smoking = yes_no("Passive Smoking")

    air_pollution_index = st.slider(
        "Air Pollution Index",
        0,
        500,
        100
    )

    occupational_exposure = yes_no(
        "Occupational Exposure"
    )

    radon_exposure = yes_no(
        "Radon Exposure"
    )

    family_history_cancer = yes_no(
        "Family History of Cancer"
    )

with col2:

    asthma = yes_no("Asthma")

    previous_tb = yes_no("Previous TB")

    chronic_cough = yes_no("Chronic Cough")

    chest_pain = yes_no("Chest Pain")

    shortness_of_breath = yes_no(
        "Shortness of Breath"
    )

    fatigue = yes_no("Fatigue")

    bmi = st.number_input(
        "BMI",
        10.0,
        50.0,
        24.5
    )

    oxygen_saturation = st.slider(
        "Oxygen Saturation",
        70,
        100,
        97
    )

    fev1_x10 = st.slider(
        "FEV1",
        10,
        120,
        80
    )

    crp_level = st.number_input(
        "CRP Level",
        0.0,
        20.0,
        1.0
    )

    xray_abnormal = yes_no(
        "X-Ray Abnormal"
    )

    exercise_hours_per_week = st.slider(
        "Exercise Hours / Week",
        0,
        20,
        3
    )

    diet_quality = quality_input(
        "Diet Quality"
    )

    alcohol_units_per_week = st.slider(
        "Alcohol Units / Week",
        0,
        30,
        2
    )

    healthcare_access = quality_input(
        "Healthcare Access"
    )

st.divider()

predict = st.button(
    "Predict COPD Risk",
    use_container_width=True
)

# ===============================================
# PREDICTION
# ===============================================

if predict:

    patient = pd.DataFrame([{

        "age": age,
        "gender": gender,
        "education_years": education_years,
        "income_level": income_level,
        "smoker": smoker,
        "smoking_years": smoking_years,
        "cigarettes_per_day": cigarettes_per_day,
        "pack_years": pack_years,
        "passive_smoking": passive_smoking,
        "air_pollution_index": air_pollution_index,
        "occupational_exposure": occupational_exposure,
        "radon_exposure": radon_exposure,
        "family_history_cancer": family_history_cancer,
        "asthma": asthma,
        "previous_tb": previous_tb,
        "chronic_cough": chronic_cough,
        "chest_pain": chest_pain,
        "shortness_of_breath": shortness_of_breath,
        "fatigue": fatigue,
        "bmi": bmi,
        "oxygen_saturation": oxygen_saturation,
        "fev1_x10": fev1_x10,
        "crp_level": crp_level,
        "xray_abnormal": xray_abnormal,
        "exercise_hours_per_week": exercise_hours_per_week,
        "diet_quality": diet_quality,
        "alcohol_units_per_week": alcohol_units_per_week,
        "healthcare_access": healthcare_access

    }])

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0]

    confidence = max(probability) * 100

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ High Risk of COPD")

    else:

        st.success("✅ Low Risk of COPD")

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(float(confidence) / 100)

    st.divider()

    st.subheader("Risk Factors")

    if age >= 55:
        st.write("• Advanced age")

    if smoker:
        st.write("• Current smoker")

    if pack_years >= 20:
        st.write("• High smoking exposure")

    if chronic_cough:
        st.write("• Chronic cough")

    if shortness_of_breath:
        st.write("• Shortness of breath")

    if oxygen_saturation < 94:
        st.write("• Low oxygen saturation")

    if fev1_x10 < 70:
        st.write("• Reduced lung function")

    if occupational_exposure:
        st.write("• Occupational exposure")

    if air_pollution_index > 150:
        st.write("• High air pollution exposure")

    if asthma:
        st.write("• History of asthma")

    st.divider()

    st.subheader("Recommendation")

    if prediction == 1:

        st.warning("""
### Recommended Next Steps

✔ Consult a Pulmonologist

✔ Schedule Spirometry Test

✔ Stop Smoking

✔ Monitor Oxygen Saturation

✔ Regular Follow-up

✔ Follow Prescribed Medication
""")

    else:

        st.success("""
### Healthy Lifestyle Advice

✔ Continue Regular Exercise

✔ Avoid Smoking

✔ Maintain Healthy Weight

✔ Annual Lung Checkup

✔ Avoid Air Pollution

✔ Eat a Balanced Diet
""")

    st.divider()

    st.subheader("Patient Summary")

    summary = f"""
Age : {age}

Smoking : {'Yes' if smoker else 'No'}

Pack Years : {pack_years}

Chronic Cough : {'Yes' if chronic_cough else 'No'}

Shortness of Breath : {'Yes' if shortness_of_breath else 'No'}

SpO₂ : {oxygen_saturation}%

Predicted COPD Risk :
{"High Risk" if prediction == 1 else "Low Risk"}

Confidence :
{confidence:.2f}%
"""

    st.code(summary)