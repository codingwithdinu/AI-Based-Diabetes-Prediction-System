"use client";

import { useMemo, useState, type FormEvent } from "react";
import {
  Activity, ArrowRight, CheckCircle2, ChevronDown, CircleHelp,
  ClipboardList, HeartPulse, Info, LoaderCircle, Menu, ShieldCheck,
  Sparkles, Stethoscope, X,
} from "lucide-react";

type FormValues = {
  pregnancies: number; glucose: number; pressure: number; skin: number;
  insulin: number; bmi: number; pedigree: number; age: number;
};

type Result = { prediction: 0 | 1; label: string; disclaimer: string };

const initialValues: FormValues = {
  pregnancies: 0, glucose: 120, pressure: 70, skin: 20,
  insulin: 80, bmi: 25, pedigree: 0.5, age: 30,
};

const fields: { key: keyof FormValues; label: string; unit: string; min: number; max: number; step: number; hint: string }[] = [
  { key: "pregnancies", label: "Pregnancies", unit: "count", min: 0, max: 20, step: 1, hint: "Number of times pregnant" },
  { key: "glucose", label: "Glucose", unit: "mg/dL", min: 1, max: 300, step: 1, hint: "Plasma glucose concentration" },
  { key: "pressure", label: "Blood pressure", unit: "mmHg", min: 1, max: 200, step: 1, hint: "Diastolic blood pressure" },
  { key: "skin", label: "Skin thickness", unit: "mm", min: 0, max: 100, step: 1, hint: "Triceps skin fold thickness" },
  { key: "insulin", label: "Insulin", unit: "µU/mL", min: 0, max: 1000, step: 1, hint: "2-hour serum insulin" },
  { key: "bmi", label: "Body mass index", unit: "kg/m²", min: 0.1, max: 100, step: 0.1, hint: "Weight-to-height index" },
  { key: "pedigree", label: "Diabetes pedigree", unit: "index", min: 0, max: 3, step: 0.01, hint: "Family-history function value" },
  { key: "age", label: "Age", unit: "years", min: 1, max: 120, step: 1, hint: "Age in completed years" },
];

export default function Home() {
  const [values, setValues] = useState<FormValues>(initialValues);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const completed = useMemo(() => fields.filter((f) => values[f.key] !== initialValues[f.key]).length, [values]);

  function update(key: keyof FormValues, value: number) {
    setValues((current) => ({ ...current, [key]: value }));
    setResult(null);
    setError("");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    const features = [
      values.pregnancies, values.glucose, values.pressure, values.skin,
      values.insulin, values.bmi, values.pedigree, values.age,
    ];
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      const raw = await response.text();
      let data: Record<string, unknown>;
      try {
        data = raw ? JSON.parse(raw) as Record<string, unknown> : {};
      } catch {
        throw new Error("Prediction service returned an invalid response (HTTP " + response.status + "). Please retry in a moment.");
      }
      if (!response.ok) {
        throw new Error(typeof data.error === "string" ? data.error : "Prediction request failed (HTTP " + response.status + ").");
      }
      if (data.prediction !== 0 && data.prediction !== 1) {
        throw new Error("The prediction service returned an unexpected result. Please try again.");
      }
      setResult(data as unknown as Result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to connect to the prediction service.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="site-shell">
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <a className="brand" href="#home" aria-label="GlucoWise home">
          <span className="brand-symbol"><Activity size={23} strokeWidth={2.5} /></span>
          <span><strong>GlucoWise</strong><small>AI HEALTH SCREENING</small></span>
        </a>
        <div className="side-label">WORKSPACE</div>
        <a className="side-link active" href="#assessment"><ClipboardList size={18}/> Screening assessment</a>
        <a className="side-link" href="#how-it-works"><Activity size={18}/> How it works</a>
        <div className="side-bottom">
          <div className="side-shield"><ShieldCheck size={19}/></div>
          <strong>Your data, your control</strong>
          <p>Measurements are sent securely to the prediction endpoint only when you submit.</p>
          <span className="model-pill"><span/> SVM MODEL · STANDARD SCALER</span>
        </div>
        <button className="mobile-close" onClick={() => setMenuOpen(false)} aria-label="Close menu"><X/></button>
      </aside>

      <section className="main-area" id="home">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMenuOpen(true)} aria-label="Open menu"><Menu/></button>
          <div className="breadcrumb">Workspace <span>/</span> <b>Health screening</b></div>
          <div className="top-status"><span className="status-dot"/> Model service interface</div>
        </header>

        <div className="content-wrap">
          <section className="hero">
            <div className="hero-copy">
              <div className="eyebrow"><Sparkles size={14}/> PREVENTIVE HEALTH · AI INSIGHTS</div>
              <h1>Know your numbers.<br/><em>Take a healthier step.</em></h1>
              <p>A guided, machine-learning screening experience using eight health measurements. Clear inputs, understandable results, and a focus on informed conversations.</p>
              <a className="hero-cta" href="#assessment">Start assessment <ArrowRight size={16}/></a>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="orbit orbit-one"/><div className="orbit orbit-two"/>
              <div className="pulse-card"><HeartPulse size={54}/><span>Health insights</span><b>Made clearer</b></div>
              <div className="float-chip chip-top"><Activity size={15}/> AI assisted</div>
              <div className="float-chip chip-bottom"><ShieldCheck size={15}/> Privacy aware</div>
              <div className="hero-cross">+</div>
            </div>
            <div className="hero-index">01 <span>/ 03</span></div>
          </section>

          <div className="stats-row">
            <article className="stat-card"><span className="stat-icon mint"><Activity/></span><div><small>MODEL</small><strong>Support Vector Machine</strong><span>Linear classification</span></div></article>
            <article className="stat-card"><span className="stat-icon blue"><ClipboardList/></span><div><small>INPUTS</small><strong>08 indicators</strong><span>Health measurements</span></div></article>
            <article className="stat-card"><span className="stat-icon lilac"><ShieldCheck/></span><div><small>EXPERIENCE</small><strong>Private by design</strong><span>Submit only when ready</span></div></article>
          </div>

          <section className="notice">
            <span className="notice-icon"><Info size={19}/></span>
            <p><strong>Important health information</strong><br/>This educational project is not a medical device and cannot diagnose diabetes. Its result must not guide treatment or replace professional advice and appropriate laboratory testing.</p>
            <a href="#disclaimer">Read disclaimer <ArrowRight size={14}/></a>
          </section>

          <section className="assessment-section" id="assessment">
            <div className="section-heading">
              <div><div className="section-eyebrow">YOUR SCREENING</div><h2>Health assessment</h2><p>Enter the measurements below to generate a model estimate.</p></div>
              <div className="completion"><span>{completed}/8</span><small>values adjusted</small></div>
            </div>

            <form className="assessment-card" onSubmit={submit}>
              <div className="form-top"><div className="form-icon"><Stethoscope size={20}/></div><div><h3>Health measurements</h3><p>Use values from a reliable measurement or medical record.</p></div><span className="required-tag">8 FIELDS</span></div>
              <div className="field-grid">
                {fields.map((field, index) => (
                  <label className="field" key={field.key}>
                    <span className="field-label">{field.label}<span>{field.unit}</span></span>
                    <span className="input-wrap">
                      <input
                        type="number" min={field.min} max={field.max} step={field.step}
                        value={values[field.key]}
                        onChange={(event) => update(field.key, Number(event.target.value))}
                        required aria-label={field.label}
                      />
                      <span className="field-index">0{index + 1}</span>
                    </span>
                    <small>{field.hint}</small>
                  </label>
                ))}
              </div>
              <div className="form-footer">
                <button type="button" className="reset-btn" onClick={() => { setValues(initialValues); setResult(null); setError(""); }}>Reset values</button>
                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? <><LoaderCircle className="spin" size={17}/> Analysing measurements...</> : <>Generate screening estimate <ArrowRight size={17}/></>}
                </button>
              </div>
            </form>

            {error && <div className="error-panel"><Info size={19}/><div><strong>Unable to generate estimate</strong><p>{error}</p></div></div>}
            {result && (
              <section className={`result-panel ${result.prediction === 1 ? "result-warn" : "result-ok"}`} aria-live="polite">
                <div className="result-icon">{result.prediction === 1 ? <Info size={24}/> : <CheckCircle2 size={24}/>}</div>
                <div className="result-content">
                  <span className="result-kicker">YOUR SCREENING RESULT</span>
                  <h3>{result.prediction === 1 ? "A possible diabetes risk was detected" : "No diabetes-related warning was detected"}</h3>
                  <p>{result.prediction === 1
                    ? "Your entered measurements were classified by the AI model in the higher-risk category. This does not mean you have diabetes. Please discuss your results with a qualified healthcare professional, who may recommend a blood glucose or HbA1c test."
                    : "Based on the measurements you entered, the AI model did not identify a diabetes-related warning. This does not guarantee that you are diabetes-free or rule out other health concerns. If you have symptoms or concerns, speak with a healthcare professional."
                  }</p>
                  <small>{result.disclaimer || "This is an educational AI screening estimate, not a medical diagnosis. Do not use it to make treatment decisions."}</small>
                </div>
              </section>
            )}
          </section>

          <section className="how-section" id="how-it-works">
            <div className="section-eyebrow">BEHIND THE EXPERIENCE</div><h2>How the estimate works</h2>
            <div className="steps-grid">
              <article><span>01</span><h3>Enter measurements</h3><p>Provide the eight values used by the original Pima dataset model.</p></article>
              <article><span>02</span><h3>Standardise inputs</h3><p>The saved StandardScaler transforms values in the same feature order.</p></article>
              <article><span>03</span><h3>Generate estimate</h3><p>The saved SVM model returns a binary screening classification.</p></article>
            </div>
          </section>

          <footer className="footer" id="disclaimer">
            <div className="footer-brand"><span className="brand-symbol"><Activity size={18}/></span><strong>GlucoWise</strong></div>
            <p>Educational AI project · Not for clinical use</p>
            <a href="https://github.com/codingwithdinu/AI-Based-Diabetes-Prediction-System" target="_blank" rel="noreferrer">View source <ArrowRight size={13}/></a>
          </footer>
        </div>
      </section>
    </main>
  );
}
