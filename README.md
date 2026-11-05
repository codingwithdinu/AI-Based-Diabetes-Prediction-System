# GlucoWise — AI-Based Diabetes Screening

A Vercel-ready web application for an educational diabetes screening project. The frontend is built with Next.js and TypeScript; a Python Vercel Function serves predictions using the repository's existing Support Vector Machine model and fitted StandardScaler.

> **Medical disclaimer:** This is an educational demonstration, not a medical device or diagnostic service. It is not clinically validated and must not be used to diagnose diabetes, make treatment decisions, or delay medical care. Consult a qualified healthcare professional.

## Architecture

- **Frontend:** Next.js 15, React 19, TypeScript, responsive CSS.
- **API:** Python Vercel Function at `/api/predict`.
- **Model:** Existing `diabetes_model.pkl` (SVM) and `scaler.pkl`.
- **Flow:** Browser form → JSON API request → input validation → StandardScaler → SVM prediction → screening result.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Open [Vercel](https://vercel.com/new) and import `codingwithdinu/AI-Based-Diabetes-Prediction-System`.
3. Keep the project root as the Root Directory.
4. Framework preset should be detected as **Next.js**.
5. Use the default install/build settings (Vercel runs `npm run build`).
6. Deploy.

No environment variables are required for the current model.

## Run locally

Use Node.js 20+ and Python 3.12.

```bash
npm install
npm run dev
```

The Next.js interface starts at http://localhost:3000. To test the Python API locally, install dependencies and use Vercel CLI:

```bash
python -m pip install -r requirements.txt
npm install -g vercel
vercel dev
```

## Model input order

The API preserves the original notebook's exact eight-feature order:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

## Deployment notes

- Keep `diabetes_model.pkl` and `scaler.pkl` in the repository root. The Python function reads these files at runtime.
- The Python runtime uses the dependencies in `requirements.txt`. Scikit-learn is pinned to reduce pickle compatibility surprises; serialized scikit-learn objects are not guaranteed to be compatible across versions.
- Vercel Functions have a read-only filesystem except for temporary storage. This app does not write model artifacts or user data at runtime.
- The existing `app.py` remains as the optional Streamlit version; Vercel serves the Next.js frontend and Python API instead.

## Model performance

The original notebook reports 78.66% training accuracy and 77.27% test accuracy on its split. These values have not been independently re-evaluated during the web-app conversion. Accuracy alone does not establish clinical suitability.

## Privacy and safety

The prediction endpoint accepts eight numeric values, validates finite values and expected ranges, and does not persist submissions. Do not upload identifiable patient records to this educational demo.
