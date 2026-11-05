from pathlib import Path
import pickle

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f8fc; }
.hero { padding: 1.6rem 1.8rem; border-radius: 18px;
background: linear-gradient(120deg,#102b46,#146b70); color: white; margin-bottom: 1.2rem; }
.hero h1 { color: white; margin: 0 0 .4rem 0; }
.hero p { color: #d7e9ef; margin: 0; }
.notice { padding: .9rem 1rem; border-left: 4px solid #159a87;
background: #eaf6f3; border-radius: 8px; color: #173b3a; }
div.stButton > button { border-radius: 10px; min-height: 3rem; font-weight: 650;
background: #116b69; color: white; border: 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load trusted model artifacts once and resolve paths relative to this file."""
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError(
            "Missing diabetes_model.pkl or scaler.pkl. Keep both files beside app.py."
        )
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with SCALER_PATH.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

st.markdown("""
<section class="hero">
<h1>🩺 Diabetes Risk Screening</h1>
<p>A machine-learning demonstration using selected health measurements.</p>
</section>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="notice"><strong>Important:</strong> This educational tool does not diagnose '
    'diabetes or replace a qualified healthcare professional. Do not make treatment decisions '
    'based on this prediction.</div>',
    unsafe_allow_html=True,
)
st.subheader("Enter health measurements")
st.caption("Enter values from a reliable measurement or medical record.")

with st.sidebar:
    st.subheader("About this project")
    st.write("Model: Support Vector Machine (SVM) with the saved fitted scaler.")
    st.caption("For education and preliminary screening only.")

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Diastolic blood pressure (mmHg)", min_value=0.0,
                                         max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("Skin thickness (mm)", min_value=0.0,
                                         max_value=100.0, value=20.0, step=1.0)
    with right:
        insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=1000.0,
                                  value=80.0, step=1.0)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0,
                              value=0.5, step=0.01, format="%.2f")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
    submitted = st.form_submit_button("Estimate risk", use_container_width=True)

if submitted:
    values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    if glucose <= 0 or blood_pressure <= 0 or bmi <= 0:
        st.error("Glucose, blood pressure, and BMI must be greater than zero.")
    else:
        try:
            model, scaler = load_artifacts()
            features = np.asarray([values], dtype=float)
            scaled_features = scaler.transform(features)
            prediction = int(model.predict(scaled_features)[0])
            st.divider()
            st.subheader("Screening result")
            if prediction == 1:
                st.warning(
                    "The model flagged this input as higher risk. This is not a diagnosis. "
                    "Discuss the result and appropriate testing with a clinician."
                )
            elif prediction == 0:
                st.success(
                    "The model did not flag this input as higher risk. This does not rule out "
                    "diabetes; consult a clinician about medical concerns."
                )
            else:
                st.error("The model returned an unexpected class. Check the model artifacts.")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Prediction failed. Check that the model, scaler, and dependencies are compatible.")

st.divider()
st.caption("Educational project. Model predictions are not clinically validated diagnoses.")
