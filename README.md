# 🩺 AI-Based Diabetes Prediction System

An educational machine-learning application using a Support Vector Machine (SVM) and a saved StandardScaler to classify input measurements using the Pima Indians Diabetes dataset.

> **Medical disclaimer:** This is an educational screening demonstration, not a medical device or diagnostic service. It has not been established as clinically validated. Do not use its output to diagnose diabetes, change medication, or delay medical care. Consult a qualified healthcare professional for medical advice and appropriate testing.

## Features

- Streamlit interface with input validation
- Saved SVM model and fitted scaler
- Model artifacts loaded relative to the app file
- Cached model loading
- Clear explanation of results and limitations

## Tech stack

Python · Streamlit · NumPy · Pandas · scikit-learn · Support Vector Machine

## Repository structure

```text
.
├── app.py
├── diabetes.csv
├── diabetes_model.pkl
├── scaler.pkl
├── requirements.txt
└── Train_ML_Model.ipynb
```

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/codingwithdinu/AI-Based-Diabetes-Prediction-System.git
cd AI-Based-Diabetes-Prediction-System
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
2. Choose **Create app** and select this repository.
3. Set branch to `main` and main file path to `app.py`.
4. Deploy.

Keep `app.py`, `requirements.txt`, `diabetes_model.pkl`, and `scaler.pkl` in the repository root.

## Model inputs

Input order must match the order used when fitting the model and scaler:

1. Pregnancies
2. Glucose
3. Blood pressure
4. Skin thickness
5. Insulin
6. BMI
7. Diabetes pedigree function
8. Age

The app uses the repository's pickle artifacts. Only load pickle files from a trusted source: unpickling an untrusted file can execute arbitrary code.

## Model performance

The original README reports approximately 78% training accuracy and 77% testing accuracy. These figures have not been independently re-evaluated as part of this interface update. Accuracy alone does not establish clinical suitability; validation on representative populations and metrics such as sensitivity, specificity, and calibration are important.

## Limitations

- Dataset-level performance may not generalize to other populations.
- Input measurements and missing-value conventions affect predictions.
- The output is a model classification, not a calibrated probability.
- This application is not intended for clinical decision-making.
