from pathlib import Path
import pickle

import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

st.set_page_config(page_title="GlucoWise | Diabetes Screening", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#15243a; --muted:#6b7a90; --teal:#087f8c; --mint:#dff5ef; --line:#e6edf3; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#f4f7fb; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(244,247,251,.88); }
.block-container { max-width:1200px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#10263b; }
[data-testid="stSidebar"] * { color:#e7f1f7 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b8c9d7 !important; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:1.4rem; }
.brand-mark { display:grid; place-items:center; width:44px; height:44px; border-radius:14px;
background:linear-gradient(135deg,#0b948e,#34b6a0); color:white; font-size:22px; box-shadow:0 8px 20px #0b948e33; }
.brand-name { font:800 1.08rem 'Manrope',sans-serif; color:#18324b; letter-spacing:-.4px; }
.brand-sub { font-size:.72rem; color:#8594a8; margin-top:1px; }
.hero { position:relative; overflow:hidden; border-radius:24px; padding:2.2rem 2.4rem;
background:linear-gradient(112deg,#10283f 0%,#124f60 58%,#0c827e 100%); color:white;
box-shadow:0 18px 45px rgba(16,40,63,.13); margin-bottom:1.3rem; }
.hero:after { content:'+'; position:absolute; right:8%; top:-90px; font-size:300px; line-height:1;
color:rgba(255,255,255,.045); font-weight:200; }
.eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:.7rem; font-weight:700; color:#9ce4d7; }
.hero h1 { font:800 clamp(1.8rem,4vw,2.65rem) 'Manrope',sans-serif; color:white; letter-spacing:-1.3px; margin:.55rem 0 .55rem; }
.hero p { color:#d1e4e9; font-size:1rem; max-width:620px; margin:0; line-height:1.7; }
.hero-tag { display:inline-flex; align-items:center; gap:7px; border:1px solid #ffffff35;
background:#ffffff12; color:#e2f6f1; border-radius:999px; padding:7px 12px; font-size:.75rem; margin-top:1.2rem; }
.section-kicker { font-size:.72rem; font-weight:700; color:#0b8b89; letter-spacing:1.6px; text-transform:uppercase; margin-bottom:.25rem; }
.section-heading { font:800 1.35rem 'Manrope',sans-serif; color:#18324b; letter-spacing:-.5px; margin-bottom:.25rem; }
.section-desc { color:#75849a; font-size:.9rem; margin-bottom:1rem; }
.metric { background:white; border:1px solid var(--line); border-radius:16px; padding:16px 18px; min-height:92px; }
.metric-label { color:#8491a2; font-size:.74rem; font-weight:600; }
.metric-value { color:#17334c; font:800 1.3rem 'Manrope',sans-serif; margin-top:5px; }
.metric-icon { float:right; font-size:1.25rem; background:#e9f7f4; border-radius:10px; padding:6px; }
.panel { background:white; border:1px solid var(--line); border-radius:20px; padding:1.35rem 1.45rem; margin:.45rem 0 1rem; }
.panel-title { font:800 1rem 'Manrope',sans-serif; color:#1b354e; margin-bottom:.25rem; }
.panel-sub { color:#8390a1; font-size:.8rem; margin-bottom:1rem; }
.notice { display:flex; gap:12px; align-items:flex-start; background:#fff9e9;
border:1px solid #f5e7bd; color:#765b22; padding:13px 16px; border-radius:14px; margin:1rem 0 1.4rem; font-size:.83rem; line-height:1.55; }
.notice strong { color:#634713; }
div[data-testid="stNumberInput"] label { color:#344b62; font-size:.82rem; font-weight:600; }
div[data-testid="stNumberInput"] input { border-radius:10px; }
.stNumberInput [data-testid="stWidgetLabel"] p { font-weight:600; }
div.stButton > button, div.stFormSubmitButton > button { border:0; border-radius:12px;
background:linear-gradient(100deg,#087f8c,#13a18f); color:white; min-height:3.1rem;
font-weight:700; box-shadow:0 8px 18px rgba(8,127,140,.16); transition:transform .15s, box-shadow .15s; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { color:white;
transform:translateY(-1px); box-shadow:0 12px 22px rgba(8,127,140,.22); }
.result-card { border-radius:18px; padding:1.35rem 1.5rem; margin-top:1rem; }
.result-high { background:#fff3f0; border:1px solid #f5d4cb; }
.result-low { background:#eaf8f2; border:1px solid #ccebdd; }
.result-title { font:800 1.15rem 'Manrope',sans-serif; margin-bottom:.35rem; }
.result-high .result-title { color:#a54535; }
.result-low .result-title { color:#167257; }
.result-copy { color:#5f7082; font-size:.88rem; line-height:1.6; }
.footer { color:#93a0ae; font-size:.75rem; text-align:center; padding:1.2rem 0 .3rem; }
hr { border-color:#e8eef3 !important; }
@media(max-width:700px) { .hero { padding:1.5rem; } .panel { padding:1rem; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError("Model files not found. Keep diabetes_model.pkl and scaler.pkl beside app.py.")
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)
    return model, scaler

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✚</div><div><div class="brand-name" style="color:white">GlucoWise</div><div class="brand-sub">AI HEALTH SCREENING</div></div></div>', unsafe_allow_html=True)
    st.markdown("### Your screening workspace")
    st.write("Enter the eight health measurements used by the trained SVM model.")
    st.markdown("---")
    st.markdown("**MODEL**")
    st.caption("Support Vector Machine")
    st.markdown("**PREPROCESSING**")
    st.caption("StandardScaler")
    st.markdown("---")
    st.caption("Educational prototype · Not a clinical diagnostic tool")

st.markdown("""
<div class="brand"><div class="brand-mark">✚</div><div><div class="brand-name">GlucoWise</div><div class="brand-sub">PERSONAL HEALTH INSIGHT PLATFORM</div></div></div>
<section class="hero">
  <div class="eyebrow">AI-powered preventive health</div>
  <h1>Understand your health.<br>Take a proactive step.</h1>
  <p>Explore a machine-learning screening estimate using key health measurements.
  A simple, guided experience designed to make your project approachable and clear.</p>
  <div class="hero-tag">✦ &nbsp; SVM-based screening model &nbsp; · &nbsp; 8 health indicators</div>
</section>
""", unsafe_allow_html=True)

m1,m2,m3=st.columns(3)
for col,icon,label,value in [
    (m1,"🧠","ML algorithm","Support Vector Machine"),
    (m2,"📋","Input indicators","08 parameters"),
    (m3,"🛡️","Purpose","Early risk awareness"),
]:
    with col:
        st.markdown(f'<div class="metric"><span class="metric-icon">{icon}</span><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)

st.markdown("""
<div class="notice"><div style="font-size:1.15rem">ⓘ</div><div><strong>Health information notice</strong><br>
This application is an educational machine-learning demonstration, not a medical device.
Its output is not a diagnosis and must not guide treatment decisions. Speak with a qualified
healthcare professional about symptoms, risk factors, or testing.</div></div>
""",unsafe_allow_html=True)

st.markdown('<div class="section-kicker">Screening assessment</div><div class="section-heading">Your health measurements</div><div class="section-desc">Fill in the fields below using values from a reliable measurement or medical record.</div>',unsafe_allow_html=True)

with st.form("screening_form"):
    left,right=st.columns(2,gap="large")
    with left:
        st.markdown('<div class="panel-title">01 · Personal & metabolic</div><div class="panel-sub">Body and glucose-related measurements</div>',unsafe_allow_html=True)
        pregnancies=st.number_input("Pregnancies",min_value=0,max_value=20,value=0,step=1,help="Number of times pregnant.")
        glucose=st.number_input("Glucose (mg/dL)",min_value=0.0,max_value=300.0,value=120.0,step=1.0)
        blood_pressure=st.number_input("Diastolic blood pressure (mmHg)",min_value=0.0,max_value=200.0,value=70.0,step=1.0)
        skin=st.number_input("Skin thickness (mm)",min_value=0.0,max_value=100.0,value=20.0,step=1.0)
    with right:
        st.markdown('<div class="panel-title">02 · Body & family indicators</div><div class="panel-sub">Additional model input measurements</div>',unsafe_allow_html=True)
        insulin=st.number_input("Insulin (µU/mL)",min_value=0.0,max_value=1000.0,value=80.0,step=1.0)
        bmi=st.number_input("BMI (kg/m²)",min_value=0.0,max_value=100.0,value=25.0,step=0.1)
        dpf=st.number_input("Diabetes pedigree function",min_value=0.0,max_value=3.0,value=0.5,step=0.01,format="%.2f")
        age=st.number_input("Age (years)",min_value=1,max_value=120,value=30,step=1)
    st.markdown("---")
    submitted=st.form_submit_button("✦  Generate screening estimate",use_container_width=True)

if submitted:
    values=[pregnancies,glucose,blood_pressure,skin,insulin,bmi,dpf,age]
    if glucose<=0 or blood_pressure<=0 or bmi<=0:
        st.error("Please enter values greater than zero for glucose, blood pressure, and BMI.")
    else:
        try:
            model,scaler=load_artifacts()
            matrix=np.asarray([values],dtype=float)
            transformed=scaler.transform(matrix)
            prediction=int(model.predict(transformed)[0])
            st.markdown('<div class="section-kicker" style="margin-top:1.5rem">Your assessment</div><div class="section-heading">Screening estimate</div>',unsafe_allow_html=True)
            if prediction==1:
                st.markdown('<div class="result-card result-high"><div class="result-title">⚠️ The model flagged this input for follow-up</div><div class="result-copy">This model output is not a diagnosis. Consider discussing your health measurements and appropriate laboratory testing with a qualified healthcare professional.</div></div>',unsafe_allow_html=True)
            elif prediction==0:
                st.markdown('<div class="result-card result-low"><div class="result-title">✓ The model did not flag this input</div><div class="result-copy">This output does not rule out diabetes or other health concerns. Consult a healthcare professional when you have symptoms, risk factors, or questions.</div></div>',unsafe_allow_html=True)
            else:
                st.error("The model returned an unsupported class. Please check the model artifacts.")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The estimate could not be generated. Check that the saved model, scaler, and dependencies are compatible.")

st.markdown('<div class="footer">GLUCOWISE · EDUCATIONAL AI PROJECT · NOT FOR CLINICAL USE</div>',unsafe_allow_html=True)
