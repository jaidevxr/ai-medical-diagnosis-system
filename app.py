"""
app.py
--------------------------------------------------------------------
Streamlit Application for the 20-Disease AI Medical Diagnosis System.

Features an authentic Ada Health Interactive Symptom Assessment Engine:
  Step 1: Patient Profile & Risk Factors
  Step 2: Chief Complaint & Symptom Intake
  Step 3: Targeted Follow-up Medical Questions
  Step 4: Ada Differential Assessment & Triage Report (with SHAP)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap

import preprocessing

# ---------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Ada AI Health Assessment & 20-Disease Diagnostic Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# CUSTOM STYLING (Ada Health UI Theme)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    
    .ada-header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
        padding: 26px;
        border-radius: 16px;
        border: 1px solid #4f46e5;
        box-shadow: 0 12px 30px -5px rgba(79, 70, 229, 0.3);
        margin-bottom: 25px;
    }
    
    .ada-header-title {
        color: #a5b4fc;
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    
    .ada-header-subtitle {
        color: #e0e7ff;
        font-size: 1.1rem;
    }

    .ada-step-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }

    .ada-badge-emergency {
        background-color: #dc2626;
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }

    .ada-badge-doctor {
        background-color: #d97706;
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }

    .ada-badge-selfcare {
        background-color: #059669;
        color: white;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .risk-badge-low {
        background-color: #059669;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-moderate {
        background-color: #d97706;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-badge-high {
        background-color: #dc2626;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .critical-alert {
        background-color: #7f1d1d;
        border-left: 6px solid #ef4444;
        padding: 18px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DISEASE_CONFIG = {
    "fever": {"name": "Fever & Viral Flu", "icon": "🤒", "color": "#ef4444", "category": "Everyday Acute"},
    "malaria": {"name": "Malaria", "icon": "🦟", "color": "#f97316", "category": "Everyday Acute"},
    "typhoid": {"name": "Typhoid Fever", "icon": "🦠", "color": "#eab308", "category": "Everyday Acute"},
    "dengue": {"name": "Dengue Fever", "icon": "🩸", "color": "#ec4899", "category": "Everyday Acute"},
    "cold": {"name": "Common Cold / URI", "icon": "🤧", "color": "#06b6d4", "category": "Everyday Acute"},
    "gastro": {"name": "Gastroenteritis", "icon": "🫄", "color": "#10b981", "category": "Everyday Acute"},
    
    "diabetes": {"name": "Diabetes Mellitus", "icon": "🩸", "color": "#ef4444", "category": "Chronic & Metabolic"},
    "hypertension": {"name": "Hypertension", "icon": "🩸", "color": "#8b5cf6", "category": "Chronic & Metabolic"},
    "anemia": {"name": "Anemia", "icon": "🩸", "color": "#a855f7", "category": "Chronic & Metabolic"},
    "thyroid": {"name": "Thyroid Disorder", "icon": "🦋", "color": "#64748b", "category": "Chronic & Metabolic"},
    "dementia": {"name": "Alzheimer's & Dementia", "icon": "⚡", "color": "#64748b", "category": "Chronic & Metabolic"},
    
    "heart": {"name": "Coronary Heart Disease", "icon": "❤️", "color": "#f97316", "category": "Severe Organic"},
    "kidney": {"name": "Chronic Kidney Disease", "icon": "🧪", "color": "#eab308", "category": "Severe Organic"},
    "liver": {"name": "Hepatic Liver Disease", "icon": "🫀", "color": "#10b981", "category": "Severe Organic"},
    "pneumonia": {"name": "Pneumonia", "icon": "🫁", "color": "#06b6d4", "category": "Severe Organic"},
    "asthma": {"name": "Asthma", "icon": "🫁", "color": "#38bdf8", "category": "Severe Organic"},
    "cancer": {"name": "Oncology / Tumor Risk", "icon": "🎗️", "color": "#ec4899", "category": "Severe Organic"},

    "stroke": {"name": "Stroke Risk", "icon": "🧠", "color": "#a855f7", "category": "CRITICAL EMERGENCY"},
    "sepsis": {"name": "Sepsis & Septic Shock", "icon": "🦠", "color": "#dc2626", "category": "CRITICAL EMERGENCY"},
    "hypertensive_crisis": {"name": "Hypertensive Crisis", "icon": "💥", "color": "#ef4444", "category": "CRITICAL EMERGENCY"},
}


# ---------------------------------------------------------------
# CACHED ARTIFACT LOADERS
# ---------------------------------------------------------------
@st.cache_resource
def load_disease_artifacts(disease_key):
    """Load model, preprocessor, feature names, explainer, and metadata for a disease."""
    disease_dir = os.path.join("models", disease_key)
    
    if not os.path.exists(disease_dir):
        if disease_key == "diabetes" and os.path.exists("models/best_model.joblib"):
            disease_dir = "models"
        else:
            return None, None, None, None, None

    model_path = os.path.join(disease_dir, "best_model.joblib")
    prep_path = os.path.join(disease_dir, "preprocessor.joblib")
    feat_path = os.path.join(disease_dir, "feature_columns.joblib")
    meta_path = os.path.join(disease_dir, "model_metadata.joblib")
    shap_path = os.path.join(disease_dir, "shap_explainer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        return None, None, None, None, None

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    feature_names = joblib.load(feat_path) if os.path.exists(feat_path) else None
    metadata = joblib.load(meta_path) if os.path.exists(meta_path) else None
    explainer = joblib.load(shap_path) if os.path.exists(shap_path) else None

    return model, preprocessor, feature_names, explainer, metadata


@st.cache_data
def load_summary_analytics():
    """Load multi-disease model benchmark summaries."""
    summary_path = "models/all_diseases_summary.csv"
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    return None


def predict_disease_risk(disease_key, df_raw):
    """Transforms raw patient dataframe and computes risk probability and classification."""
    model, preprocessor, feature_names, explainer, metadata = load_disease_artifacts(disease_key)
    if model is None or preprocessor is None:
        return None

    df_engineered = preprocessing.engineer_disease_features(df_raw, disease_key)
    transformed_X = preprocessor.transform(df_engineered)
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(transformed_X)[0, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(transformed_X)[0]
        prob = float(1 / (1 + np.exp(-d)))
    else:
        prob = float(model.predict(transformed_X)[0])

    if prob < 0.35:
        category = "Low Risk"
        badge_cls = "risk-badge-low"
    elif prob < 0.65:
        category = "Moderate Risk"
        badge_cls = "risk-badge-moderate"
    else:
        category = "High Risk"
        badge_cls = "risk-badge-high"

    return {
        "disease_key": disease_key,
        "disease_name": DISEASE_CONFIG[disease_key]["name"],
        "icon": DISEASE_CONFIG[disease_key]["icon"],
        "category_tier": DISEASE_CONFIG[disease_key]["category"],
        "probability": float(prob),
        "percentage": float(prob * 100),
        "category": category,
        "badge_cls": badge_cls,
        "transformed_X": transformed_X,
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "feature_names": feature_names,
        "metadata": metadata,
    }


# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
st.sidebar.title("🩺 AI Medical Diagnostic Suite")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🤖 Ada Interactive AI Health Assessment",
        "Home Overview",
        "🏥 Universal 20-Disease Health Scanner",
        "🤒 Everyday Fever, Flu & Infection Diagnostic",
        "🩸 Metabolic & Chronic Disease Diagnostic",
        "❤️ Cardiac, Organ & Cancer Diagnostic",
        "🧠 Critical Emergency & Shock Diagnostic",
        "📊 Model Analytics & Benchmarks",
        "ℹ️ Architecture & Defense",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("🛡️ **Zero Data Leakage Pipeline**: Imputation, scaling & encoding strictly fitted per fold/set.")


# ---------------------------------------------------------------
# PAGE: ADA INTERACTIVE AI HEALTH ASSESSMENT (ADA HEALTH APP FLOW)
# ---------------------------------------------------------------
if page == "🤖 Ada Interactive AI Health Assessment":
    st.markdown(
        """
        <div class="ada-header-card">
            <div class="ada-header-title">🤖 Ada Interactive AI Health Assessment</div>
            <div class="ada-header-subtitle">
                Complete a guided 4-step medical symptom consultation. Ada AI automatically checks <b>20 disease possibilities</b> 
                and provides a clinical triage report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "ada_step" not in st.session_state:
        st.session_state.ada_step = 1

    # STEP PROGRESS INDICATOR
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("🔴 **1. Profile & Risks**" if st.session_state.ada_step == 1 else "⚪ 1. Profile")
    p2.markdown("🔴 **2. Chief Symptom**" if st.session_state.ada_step == 2 else "⚪ 2. Chief Symptom")
    p3.markdown("🔴 **3. Follow-up Questions**" if st.session_state.ada_step == 3 else "⚪ 3. Questions")
    p4.markdown("🔴 **4. Ada Health Report**" if st.session_state.ada_step == 4 else "⚪ 4. Report")
    st.markdown("---")

    # STEP 1: PATIENT PROFILE & RISK FACTORS
    if st.session_state.ada_step == 1:
        st.markdown("<div class='ada-step-box'>", unsafe_allow_html=True)
        st.subheader("Step 1 of 4: Patient Profile & Health Background")
        
        target_person = st.radio("Who is this assessment for?", ["Myself", "Someone Else"], horizontal=True)
        
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            age = st.number_input("Age (Years)", 1, 100, 34)
            sex = st.selectbox("Biological Sex", ["Female", "Male"])
        with c_p2:
            rf_hbp = st.checkbox("High Blood Pressure")
            rf_hchol = st.checkbox("High Cholesterol")
            rf_diab = st.checkbox("Diabetes Mellitus")
            rf_smoker = st.checkbox("Active Smoker")
            rf_heart = st.checkbox("Heart Disease History")

        if st.button("Continue to Symptoms ➔", type="primary", use_container_width=True):
            st.session_state.ada_age = age
            st.session_state.ada_sex = sex
            st.session_state.ada_rf_hbp = rf_hbp
            st.session_state.ada_rf_hchol = rf_hchol
            st.session_state.ada_rf_diab = rf_diab
            st.session_state.ada_rf_smoker = rf_smoker
            st.session_state.ada_rf_heart = rf_heart
            st.session_state.ada_step = 2
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # STEP 2: CHIEF COMPLAINT / MAIN SYMPTOM
    elif st.session_state.ada_step == 2:
        st.markdown("<div class='ada-step-box'>", unsafe_allow_html=True)
        st.subheader("Step 2 of 4: What is your primary symptom?")
        st.write("Select what is bothering you the most today:")

        chief_symptom = st.selectbox(
            "Primary Complaint",
            [
                "🤒 Fever / High Body Temperature",
                "🥶 Cold Chills & Shivering Paroxysms",
                "🤕 Headaches / Dizziness / Memory Issues",
                "🗣️ Sore Throat, Nasal Congestion & Cold",
                "😮‍💨 Cough, Shortness of Breath & Wheezing",
                "🤢 Nausea, Vomiting, Diarrhea & Stomach Cramps",
                "🫀 Chest Pain / Pressure / Palpitations",
                "🦴 Severe Joint Pain / Body Aches",
                "🩸 Unexplained Fatigue, Pallor or Low Platelet Symptoms",
            ]
        )

        symptom_duration = st.select_slider(
            "How long have you had this symptom?",
            options=["Less than 24 hours", "1 - 3 Days", "4 - 7 Days", "More than 1 week"]
        )

        symptom_notes = st.text_input("Describe any additional symptoms in your own words (Optional):", "")

        c_b1, c_b2 = st.columns([1, 1])
        with c_b1:
            if st.button("⬅️ Back"):
                st.session_state.ada_step = 1
                st.rerun()
        with c_b2:
            if st.button("Continue to Assessment Questions ➔", type="primary", use_container_width=True):
                st.session_state.ada_chief = chief_symptom
                st.session_state.ada_duration = symptom_duration
                st.session_state.ada_notes = symptom_notes
                st.session_state.ada_step = 3
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # STEP 3: TARGETED MEDICAL FOLLOW-UP QUESTIONS
    elif st.session_state.ada_step == 3:
        st.markdown("<div class='ada-step-box'>", unsafe_allow_html=True)
        st.subheader("Step 3 of 4: Targeted Clinical Follow-up Questions")
        st.write("Ada AI is refining the differential diagnosis based on your answers:")

        q1 = st.radio("1. Are you experiencing sudden high fever with intense shivering or cold chills?", ["Yes", "No", "Unsure"], horizontal=True)
        q2 = st.radio("2. Do you have severe pain behind your eyes or intense bone and muscle aching?", ["Yes", "No", "Unsure"], horizontal=True)
        q3 = st.radio("3. Are you suffering from frequent nausea, vomiting, or diarrhea episodes?", ["Yes", "No", "Unsure"], horizontal=True)
        q4 = st.radio("4. Have you noticed shortness of breath, chest tightness, or wheezing?", ["Yes", "No", "Unsure"], horizontal=True)
        q5 = st.radio("5. Have you experienced any severe dizziness, confusion, or memory loss?", ["Yes", "No", "Unsure"], horizontal=True)

        c_b1, c_b2 = st.columns([1, 1])
        with c_b1:
            if st.button("⬅️ Back"):
                st.session_state.ada_step = 2
                st.rerun()
        with c_b2:
            if st.button("Generate Ada Clinical Assessment Report 🚀", type="primary", use_container_width=True):
                st.session_state.ada_q1 = q1
                st.session_state.ada_q2 = q2
                st.session_state.ada_q3 = q3
                st.session_state.ada_q4 = q4
                st.session_state.ada_q5 = q5
                st.session_state.ada_step = 4
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # STEP 4: ADA DIFFERENTIAL ASSESSMENT REPORT
    elif st.session_state.ada_step == 4:
        st.subheader("Step 4 of 4: Ada AI Clinical Health Assessment Report")

        age = st.session_state.get("ada_age", 34)
        sex = st.session_state.get("ada_sex", "Male")
        sex_val = 1 if sex == "Male" else 0
        
        rf_hbp = st.session_state.get("ada_rf_hbp", False)
        rf_hchol = st.session_state.get("ada_rf_hchol", False)
        rf_diab = st.session_state.get("ada_rf_diab", False)
        rf_smoker = st.session_state.get("ada_rf_smoker", False)
        rf_heart = st.session_state.get("ada_rf_heart", False)

        chief = st.session_state.get("ada_chief", "")
        notes = st.session_state.get("ada_notes", "").lower()
        full_text = chief.lower() + " " + notes

        q1 = st.session_state.get("ada_q1", "No") == "Yes"
        q2 = st.session_state.get("ada_q2", "No") == "Yes"
        q3 = st.session_state.get("ada_q3", "No") == "Yes"
        q4 = st.session_state.get("ada_q4", "No") == "Yes"
        q5 = st.session_state.get("ada_q5", "No") == "Yes"

        has_fever = "fever" in full_text or q1
        has_chills = "chills" in full_text or q1
        has_joint_eye = "joint" in full_text or q2
        has_gastro = "nausea" in full_text or "vomit" in full_text or q3
        has_resp = "breath" in full_text or "cough" in full_text or q4
        has_neuro = "dizziness" in full_text or "memory" in full_text or q5
        has_chest = "chest pain" in full_text or rf_heart

        cdc_age_bin = int(np.clip(1 + (age - 18) // 5, 1, 13))

        inputs_ada = {
            "fever": pd.DataFrame([{"BodyTemp": 39.2 if has_fever else 37.0, "Chills": 1 if has_chills else 0, "BodyAches": 1 if has_joint_eye else 0, "FatigueLevel": 3 if has_fever else 1, "Headache": 1 if has_fever else 0, "Cough": 1 if has_resp else 0, "DurationDays": 3}]),
            "malaria": pd.DataFrame([{"TempSpike": 39.5 if has_fever else 37.0, "ShiveringParoxysm": 1 if has_chills else 0, "SweatingStage": 1 if has_chills else 0, "PlateletCount": 70 if has_chills else 220, "Jaundice": 1 if "yellow" in full_text else 0, "Splenomegaly": 0}]),
            "typhoid": pd.DataFrame([{"StepladderFever": 1 if has_fever else 0, "FeverDuration": 6, "AbdominalPain": 1 if has_gastro else 0, "RelativeBradycardia": 1 if has_fever else 0, "RoseSpots": 0, "WBC_Count": 11.5}]),
            "dengue": pd.DataFrame([{"HighFever": 39.6 if has_fever else 37.0, "RetroOrbitalPain": 1 if has_joint_eye else 0, "SevereJointPain": 1 if has_joint_eye else 0, "PlateletCount": 60 if has_joint_eye else 180, "PetechiaeRash": 1 if has_joint_eye else 0, "Hematocrit": 46.0}]),
            "cold": pd.DataFrame([{"Rhinorrhea": 1 if has_resp else 0, "SoreThroat": 1 if has_resp else 0, "Sneezing": 1 if has_resp else 0, "NasalCongestion": 1 if has_resp else 0, "MildFever": 37.8 if has_fever else 36.8}]),
            "gastro": pd.DataFrame([{"Nausea": 1 if has_gastro else 0, "VomitingEpisodes": 3 if has_gastro else 0, "DiarrheaEpisodes": 4 if has_gastro else 0, "AbdominalCramps": 1 if has_gastro else 0, "DehydrationScore": 2 if has_gastro else 0}]),
            "anemia": pd.DataFrame([{"Hemoglobin": 9.0 if has_neuro else 13.5, "RBC_Count": 3.4 if has_neuro else 4.5, "Ferritin": 15 if has_neuro else 80, "Fatigue": 3 if has_neuro else 1, "Pallor": 1 if has_neuro else 0}]),
            "thyroid": pd.DataFrame([{"TSH": 6.8 if has_neuro else 2.1, "Free_T3": 2.1, "Free_T4": 0.8, "WeightChange": 1, "RestingHR": 72}]),
            "diabetes": pd.DataFrame([{"HighBP": 1 if rf_hbp else 0, "HighChol": 1 if rf_hchol else 0, "CholCheck": 1, "BMI": 29.0, "Smoker": 1 if rf_smoker else 0, "Stroke": 0, "HeartDiseaseorAttack": 1 if rf_heart else 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 2, "PhysHlth": 2, "DiffWalk": 0, "Sex": sex_val, "Age": cdc_age_bin, "Education": 5, "Income": 7}]),
            "hypertension": pd.DataFrame([{"Age": age, "SystolicBP": 160 if rf_hbp else 125, "DiastolicBP": 95 if rf_hbp else 82, "BMI": 28.5, "SodiumIntake": 2, "PhysicalInactivity": 0, "FamilyHistory": 1, "AlcoholUse": 0, "StressLevel": 3}]),
            "dementia": pd.DataFrame([{"Age": age, "MMSE_Score": 18 if has_neuro else 29, "CDR_Scale": 1.0 if has_neuro else 0.0, "FunctionalAssessment": 6 if has_neuro else 10, "MemoryLossScore": 1 if has_neuro else 0, "BehavioralProblems": 0, "EducationYears": 14, "APOE4_Allele": 0}]),
            "heart": pd.DataFrame([{"Age": age, "Sex": sex_val, "ChestPainType": 3 if has_chest else 0, "RestingBP": 145 if rf_hbp else 120, "Cholesterol": 240 if rf_hchol else 190, "FastingBS": 1 if rf_diab else 0, "RestingECG": 1 if has_chest else 0, "MaxHR": 145, "ExerciseAngina": 1 if has_chest else 0, "Oldpeak": 1.5 if has_chest else 0.0, "ST_Slope": 2, "ca": 0, "thal": 2}]),
            "kidney": pd.DataFrame([{"Age": age, "BloodPressure": 140 if rf_hbp else 120, "SpecificGravity": 1.015, "Albumin": 2 if rf_hbp else 0, "Sugar": 1 if rf_diab else 0, "BloodGlucoseRandom": 180 if rf_diab else 100, "BloodUrea": 55 if rf_hbp else 30, "SerumCreatinine": 2.2 if rf_hbp else 0.9, "Hemoglobin": 10.5 if has_neuro else 13.0, "Hypertension": 1 if rf_hbp else 0, "DiabetesMellitus": 1 if rf_diab else 0}]),
            "liver": pd.DataFrame([{"Age": age, "Gender": sex_val, "TotalBilirubin": 2.8 if "yellow" in full_text else 0.9, "DirectBilirubin": 1.0, "AlkalinePhosphatase": 230, "AlamineAminotransferase": 65, "AspartateAminotransferase": 75, "TotalProteins": 6.5, "Albumin": 3.2, "AlbuminAndGlobulinRatio": 0.9}]),
            "pneumonia": pd.DataFrame([{"Age": age, "SpO2": 91.0 if has_resp else 98.0, "RespiratoryRate": 26 if has_resp else 18, "FeverTemp": 38.8 if has_fever else 37.0, "WBC_Count": 15.5 if has_fever else 7.5, "DyspneaSeverity": 1 if has_resp else 0, "CoughType": 1 if has_resp else 0, "ChestPain": 1 if has_chest else 0, "Smoker": 1 if rf_smoker else 0}]),
            "asthma": pd.DataFrame([{"Wheezing": 1 if has_resp else 0, "PeakExpiratoryFlow": 65 if has_resp else 95, "NocturnalCough": 1 if has_resp else 0, "AllergenTrigger": 1, "ExertionalDyspnea": 1 if has_resp else 0}]),
            "cancer": pd.DataFrame([{"mean radius": 16.2 if "lump" in full_text else 14.0, "mean texture": 21.0, "mean perimeter": 105.0, "mean area": 800.0, "mean smoothness": 0.11, "mean compactness": 0.14, "mean concavity": 0.12, "mean concave points": 0.08, "mean symmetry": 0.20, "mean fractal dimension": 0.065}]),
            "stroke": pd.DataFrame([{"Gender": sex_val, "Age": age, "Hypertension": 1 if rf_hbp else 0, "HeartDisease": 1 if rf_heart else 0, "EverMarried": 1, "AvgGlucoseLevel": 200 if rf_diab else 100, "BMI": 29.0, "SmokingStatus": 1 if rf_smoker else 0}]),
            "sepsis": pd.DataFrame([{"HeartRate": 115 if (has_fever and has_chills) else 75, "SysBP": 88 if (has_fever and has_chills) else 120, "RespRate": 26 if (has_fever and has_chills) else 18, "BodyTemp": 39.2 if has_fever else 37.0, "WBC_Count": 17.5 if has_fever else 7.0, "Lactate": 4.2 if (has_fever and has_chills) else 1.1, "Platelets": 80 if has_chills else 220, "Bilirubin": 1.8, "Creatinine": 1.8}]),
            "hypertensive_crisis": pd.DataFrame([{"SystolicBP": 190 if (rf_hbp and has_chest) else 125, "DiastolicBP": 115 if (rf_hbp and has_chest) else 82, "ChestPain": 1 if has_chest else 0, "BlurredVision": 1 if has_neuro else 0, "SevereHeadache": 1 if has_fever else 0, "TargetOrganDamage": 1 if has_chest else 0}]),
        }

        results = []
        for key in DISEASE_CONFIG:
            res = predict_disease_risk(key, inputs_ada[key])
            if res:
                results.append(res)

        results.sort(key=lambda x: x["percentage"], reverse=True)
        top = results[0]

        st.markdown("<div class='ada-step-box'>", unsafe_allow_html=True)
        
        # TRIAGE BADGE
        if top["percentage"] > 65 and "CRITICAL" in top["category_tier"]:
            st.markdown("<div class='ada-badge-emergency'>🔴 Emergency Triage: Seek Immediate Emergency Care</div>", unsafe_allow_html=True)
        elif top["percentage"] > 50:
            st.markdown("<div class='ada-badge-doctor'>🟡 Medical Guidance: Doctor Consultation Recommended (within 24h)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ada-badge-selfcare'>🟢 Self Care: Home Rest & Symptom Monitoring</div>", unsafe_allow_html=True)

        st.write("")
        c_r1, c_r2 = st.columns([1.2, 1])

        with c_r1:
            st.markdown(f"### Top Suspected Condition: {top['icon']} {top['disease_name']}")
            badge_cls = top["badge_cls"]
            category_str = top["category"]
            percentage_val = top["percentage"]
            st.markdown(f"<div class='{badge_cls}'>Match Score: {percentage_val:.1f}% ({category_str})</div>", unsafe_allow_html=True)

            st.write("")
            st.write(f"**Medical Category**: {top['category_tier']}")
            st.write(f"**AI Model Baseline**: {top['metadata']['best_model_name']} (ROC-AUC: {top['metadata']['test_roc_auc']:.3f})")

            st.markdown("#### 💡 Clinical Recommendation")
            if top["percentage"] > 65:
                st.error("🚨 Go to the nearest emergency room or call emergency medical services immediately.")
            elif top["percentage"] > 40:
                st.warning("👨‍⚕️ Book an appointment with a General Physician. Request a CBC blood panel and lab tests.")
            else:
                st.success("🏠 Rest, stay hydrated, monitor your temperature, and consult a doctor if symptoms persist past 3 days.")

        with c_r2:
            st.markdown("### 📋 Ranked Differential Diagnoses")
            for r in results[:5]:
                st.write(f"**{r['icon']} {r['disease_name']}** ({r['percentage']:.1f}%)")
                st.progress(min(1.0, r['percentage'] / 100.0))

        if top["explainer"] is not None and top["transformed_X"] is not None:
            st.markdown("---")
            st.subheader(f"🔬 SHAP Clinical Attribution: Why Ada AI Suspects {top['disease_name']}")
            try:
                shap_vals = top["explainer"](top["transformed_X"])
                fig_shap, ax = plt.subplots(figsize=(8, 4))
                shap.plots.waterfall(shap_vals[0], show=False)
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"SHAP chart notice: {e}")

        st.markdown("---")
        if st.button("🔄 Start New Ada Assessment", type="primary"):
            st.session_state.ada_step = 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# PAGE: HOME OVERVIEW
# ---------------------------------------------------------------
elif page == "Home Overview":
    st.markdown(
        """
        <div class="header-card">
            <div class="header-title">🩺 20-Disease AI Medical Diagnosis Platform</div>
            <div class="header-subtitle">
                An enterprise-grade, zero-leakage clinical diagnostic platform supporting real-time risk assessment 
                across <b>20 disease domains (Everyday Normal Illnesses to Extreme Critical Emergencies)</b> with SHAP explainability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_df = load_summary_analytics()
    
    st.subheader("🌐 Covered Diagnostic Tiers")
    g1, g2, g3, g4 = st.columns(4)
    
    with g1:
        st.markdown("### 🤒 Everyday Acute Ailments")
        st.caption("Fever, Viral Flu, Malaria, Typhoid, Dengue, Common Cold, Gastroenteritis")

    with g2:
        st.markdown("### 🩸 Chronic & Metabolic")
        st.caption("Diabetes, Hypertension, Anemia, Thyroid Disorder, Dementia")

    with g3:
        st.markdown("### ❤️ Severe Organic")
        st.caption("Coronary Heart Disease, Kidney CKD, Liver Disease, Pneumonia, Asthma, Oncology")

    with g4:
        st.markdown("### 🚨 Critical Emergency")
        st.caption("Stroke Attack, Sepsis Shock, Hypertensive Crisis")

    st.markdown("---")
    
    if summary_df is not None:
        st.subheader("📈 20-Disease Model Benchmarks Summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Active Disease Models", len(summary_df))
        col_b.metric("Avg Test ROC-AUC", f"{summary_df['test_roc_auc'].mean():.3f}")
        col_c.metric("Avg Test Accuracy", f"{summary_df['test_accuracy'].mean() * 100:.1f}%")
        
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Run `python train_model.py` to view multi-disease benchmark analytics.")


# ---------------------------------------------------------------
# PAGE: UNIVERSAL 20-DISEASE SCANNER
# ---------------------------------------------------------------
elif page == "🏥 Universal 20-Disease Health Scanner":
    st.title("🏥 Universal 20-Disease Patient Health Scanner")
    st.markdown("Input patient symptoms, vitals, and lab biomarkers once to run **simultaneous risk predictions across all 20 disease models**.")

    with st.form("universal_20_scanner_form"):
        st.subheader("1. Patient Symptoms, Vitals & Lab Panel")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("🤒 Symptoms & Acute Complaints")
            body_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 38.3, step=0.1)
            chills_shivering = st.selectbox("Chills / Shivering?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            body_aches = st.selectbox("Severe Body Aches / Joint Pain?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            sore_throat = st.selectbox("Sore Throat / Runny Nose?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            nausea_vomit = st.selectbox("Nausea / Vomiting / Diarrhea?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with col2:
            st.caption("👤 Demographics & Core Vitals")
            age_years = st.number_input("Age (Years)", 1, 100, 48)
            sex_gender = st.selectbox("Gender / Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 136)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 150, 86)
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 92)

        with col3:
            st.caption("🩸 Metabolic & Blood Panel")
            glucose_val = st.number_input("Glucose Level (mg/dL)", 60, 400, 125)
            chol_val = st.number_input("Cholesterol (mg/dL)", 100, 500, 215)
            hemoglobin_val = st.number_input("Hemoglobin (g/dL)", 4.0, 18.0, 11.5, step=0.1)
            platelet_count = st.number_input("Platelets (k/µL)", 10, 600, 120)
            wbc_count = st.number_input("WBC Count (k/µL)", 1.0, 45.0, 11.0, step=0.5)

        with col4:
            st.caption("🧪 Organ & Critical Markers")
            serum_creat = st.number_input("Serum Creatinine (mg/dL)", 0.4, 15.0, 1.2, step=0.1)
            total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.3, 25.0, 1.1, step=0.1)
            spo2_val = st.number_input("SpO2 Oxygen Saturation (%)", 70, 100, 96)
            lactate_val = st.number_input("Serum Lactate (mmol/L)", 0.5, 18.0, 1.9, step=0.1)
            mmse_score = st.slider("MMSE Cognitive Score (0-30)", 8, 30, 27)

        scan_submitted = st.form_submit_button("🚀 Run Universal 20-Disease Health Scan", use_container_width=True)

    if scan_submitted:
        cdc_age_bin = int(np.clip(1 + (age_years - 18) // 5, 1, 13))

        inputs_raw = {
            "fever": pd.DataFrame([{
                "BodyTemp": body_temp, "Chills": chills_shivering, "BodyAches": body_aches,
                "FatigueLevel": 2, "Headache": 1, "Cough": sore_throat, "DurationDays": 3
            }]),
            "malaria": pd.DataFrame([{
                "TempSpike": body_temp, "ShiveringParoxysm": chills_shivering, "SweatingStage": chills_shivering,
                "PlateletCount": platelet_count, "Jaundice": 1 if total_bilirubin > 2.0 else 0, "Splenomegaly": 0
            }]),
            "typhoid": pd.DataFrame([{
                "StepladderFever": 1 if body_temp > 38.5 else 0, "FeverDuration": 6, "AbdominalPain": nausea_vomit,
                "RelativeBradycardia": 1 if heart_rate < 80 and body_temp > 38.5 else 0, "RoseSpots": 0, "WBC_Count": wbc_count
            }]),
            "dengue": pd.DataFrame([{
                "HighFever": body_temp, "RetroOrbitalPain": 1, "SevereJointPain": body_aches,
                "PlateletCount": platelet_count, "PetechiaeRash": 0, "Hematocrit": 44.0
            }]),
            "cold": pd.DataFrame([{
                "Rhinorrhea": sore_throat, "SoreThroat": sore_throat, "Sneezing": sore_throat,
                "NasalCongestion": sore_throat, "MildFever": body_temp
            }]),
            "gastro": pd.DataFrame([{
                "Nausea": nausea_vomit, "VomitingEpisodes": 3 if nausea_vomit == 1 else 0,
                "DiarrheaEpisodes": 4 if nausea_vomit == 1 else 0, "AbdominalCramps": nausea_vomit, "DehydrationScore": 1
            }]),
            "anemia": pd.DataFrame([{
                "Hemoglobin": hemoglobin_val, "RBC_Count": 3.9, "Ferritin": 25, "Fatigue": 2, "Pallor": 1 if hemoglobin_val < 11.0 else 0
            }]),
            "thyroid": pd.DataFrame([{
                "TSH": 4.5, "Free_T3": 3.0, "Free_T4": 1.1, "WeightChange": 1, "RestingHR": heart_rate
            }]),
            "diabetes": pd.DataFrame([{
                "HighBP": 1 if systolic_bp > 130 else 0, "HighChol": 1 if chol_val > 200 else 0,
                "CholCheck": 1, "BMI": 28.0, "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
                "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                "NoDocbcCost": 0, "GenHlth": 3, "MentHlth": 2, "PhysHlth": 2, "DiffWalk": 0,
                "Sex": sex_gender, "Age": cdc_age_bin, "Education": 5, "Income": 7
            }]),
            "hypertension": pd.DataFrame([{
                "Age": age_years, "SystolicBP": systolic_bp, "DiastolicBP": diastolic_bp, "BMI": 28.0,
                "SodiumIntake": 2, "PhysicalInactivity": 0, "FamilyHistory": 1, "AlcoholUse": 0, "StressLevel": 3
            }]),
            "dementia": pd.DataFrame([{
                "Age": age_years, "MMSE_Score": mmse_score, "CDR_Scale": 0.5 if mmse_score < 25 else 0.0,
                "FunctionalAssessment": 8, "MemoryLossScore": 1 if mmse_score < 26 else 0,
                "BehavioralProblems": 0, "EducationYears": 14, "APOE4_Allele": 0
            }]),
            "heart": pd.DataFrame([{
                "Age": age_years, "Sex": sex_gender, "ChestPainType": 3, "RestingBP": systolic_bp,
                "Cholesterol": chol_val, "FastingBS": 1 if glucose_val > 120 else 0, "RestingECG": 1,
                "MaxHR": heart_rate, "ExerciseAngina": 0, "Oldpeak": 1.0, "ST_Slope": 2, "ca": 0, "thal": 2
            }]),
            "kidney": pd.DataFrame([{
                "Age": age_years, "BloodPressure": systolic_bp, "SpecificGravity": 1.020,
                "Albumin": 1 if serum_creat > 1.3 else 0, "Sugar": 1 if glucose_val > 140 else 0,
                "BloodGlucoseRandom": glucose_val, "BloodUrea": 42,
                "SerumCreatinine": serum_creat, "Hemoglobin": hemoglobin_val,
                "Hypertension": 1 if systolic_bp > 130 else 0, "DiabetesMellitus": 1 if glucose_val > 140 else 0
            }]),
            "liver": pd.DataFrame([{
                "Age": age_years, "Gender": sex_gender, "TotalBilirubin": total_bilirubin,
                "DirectBilirubin": round(total_bilirubin * 0.3, 1), "AlkalinePhosphatase": 210,
                "AlamineAminotransferase": 38, "AspartateAminotransferase": 45,
                "TotalProteins": 6.8, "Albumin": 3.4, "AlbuminAndGlobulinRatio": 1.0
            }]),
            "pneumonia": pd.DataFrame([{
                "Age": age_years, "SpO2": spo2_val, "RespiratoryRate": 20, "FeverTemp": body_temp,
                "WBC_Count": wbc_count, "DyspneaSeverity": 1 if spo2_val < 95 else 0,
                "CoughType": 1, "ChestPain": 0, "Smoker": 0
            }]),
            "asthma": pd.DataFrame([{
                "Wheezing": 1 if spo2_val < 95 else 0, "PeakExpiratoryFlow": 75, "NocturnalCough": 1,
                "AllergenTrigger": 1, "ExertionalDyspnea": 1
            }]),
            "cancer": pd.DataFrame([{
                "mean radius": 14.2, "mean texture": 19.5, "mean perimeter": 92.0, "mean area": 650.0,
                "mean smoothness": 0.095, "mean compactness": 0.105, "mean concavity": 0.088,
                "mean concave points": 0.048, "mean symmetry": 0.180, "mean fractal dimension": 0.062
            }]),
            "stroke": pd.DataFrame([{
                "Gender": sex_gender, "Age": age_years, "Hypertension": 1 if systolic_bp > 130 else 0,
                "HeartDisease": 0, "EverMarried": 1, "AvgGlucoseLevel": glucose_val,
                "BMI": 28.0, "SmokingStatus": 0
            }]),
            "sepsis": pd.DataFrame([{
                "HeartRate": heart_rate, "SysBP": systolic_bp, "RespRate": 20, "BodyTemp": body_temp,
                "WBC_Count": wbc_count, "Lactate": lactate_val, "Platelets": platelet_count,
                "Bilirubin": total_bilirubin, "Creatinine": serum_creat
            }]),
            "hypertensive_crisis": pd.DataFrame([{
                "SystolicBP": systolic_bp, "DiastolicBP": diastolic_bp, "ChestPain": 0,
                "BlurredVision": 0, "SevereHeadache": 1 if systolic_bp > 160 else 0, "TargetOrganDamage": 0
            }]),
        }

        scan_results = []
        for key in DISEASE_CONFIG:
            res = predict_disease_risk(key, inputs_raw[key])
            if res:
                scan_results.append(res)

        if not scan_results:
            st.error("No disease models loaded! Execute `python train_model.py` first.")
        else:
            critical_high = [r for r in scan_results if r["category"] == "High Risk" and "CRITICAL" in r["category_tier"]]
            if critical_high:
                st.markdown(
                    f"""
                    <div class="critical-alert">
                        ⚠️ CRITICAL EMERGENCY TRIAGE ALERT: Patient is at HIGH RISK for {', '.join([c['disease_name'] for c in critical_high])}! 
                        Immediate emergency clinical intervention required!
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.subheader("📋 Comprehensive 20-Disease Patient Scorecard")
            
            avg_risk = np.mean([r["percentage"] for r in scan_results])
            high_count = sum(1 for r in scan_results if r["category"] == "High Risk")
            mod_count = sum(1 for r in scan_results if r["category"] == "Moderate Risk")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall Health Risk Index", f"{avg_risk:.1f}%")
            m2.metric("High Risk Conditions", high_count)
            m3.metric("Moderate Risk Conditions", mod_count)
            m4.metric("Total Scanned Diseases", len(scan_results))

            st.markdown("### 📊 20-Disease Risk Percentage Breakdown")
            
            df_chart = pd.DataFrame([
                {"Disease": r["disease_name"], "Risk %": r["percentage"], "Category": r["category"], "Tier": r["category_tier"]}
                for r in scan_results
            ]).sort_values("Risk %", ascending=True)
            
            fig = px.bar(
                df_chart,
                x="Risk %",
                y="Disease",
                orientation="h",
                color="Category",
                color_discrete_map={
                    "Low Risk": "#059669",
                    "Moderate Risk": "#d97706",
                    "High Risk": "#dc2626",
                },
                text="Risk %",
                range_x=[0, 100],
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(template="plotly_dark", height=650)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------
# CATEGORIZED DISEASE DIAGNOSTIC PAGES
# ---------------------------------------------------------------
elif page in ["🤒 Everyday Fever, Flu & Infection Diagnostic", "🩸 Metabolic & Chronic Disease Diagnostic", "❤️ Cardiac, Organ & Cancer Diagnostic", "🧠 Critical Emergency & Shock Diagnostic"]:
    st.title(f"🔍 {page}")
    st.markdown("Select a specific disease condition below for detailed diagnosis and SHAP explanation.")

    tier_map = {
        "🤒 Everyday Fever, Flu & Infection Diagnostic": ["fever", "malaria", "typhoid", "dengue", "cold", "gastro"],
        "🩸 Metabolic & Chronic Disease Diagnostic": ["diabetes", "hypertension", "anemia", "thyroid", "dementia"],
        "❤️ Cardiac, Organ & Cancer Diagnostic": ["heart", "kidney", "liver", "pneumonia", "asthma", "cancer"],
        "🧠 Critical Emergency & Shock Diagnostic": ["stroke", "sepsis", "hypertensive_crisis"],
    }
    
    selected_keys = tier_map[page]
    d_key = st.selectbox("Choose Disease Model to Evaluate", selected_keys, format_func=lambda k: f"{DISEASE_CONFIG[k]['icon']} {DISEASE_CONFIG[k]['name']}")
    cfg = DISEASE_CONFIG[d_key]

    model, preprocessor, feature_names, explainer, metadata = load_disease_artifacts(d_key)
    if model is None:
        st.error(f"Model artifact for {cfg['name']} missing! Please run `python train_model.py` first.")
    else:
        with st.form(f"{d_key}_form"):
            st.subheader(f"1. Clinical Parameters for {cfg['name']}")
            
            if d_key == "fever":
                temp = st.number_input("Body Temperature (°C)", 36.0, 41.5, 38.5, step=0.1)
                chills = st.selectbox("Chills / Shivering?", [0, 1])
                body_aches = st.selectbox("Severe Body Aches?", [0, 1])
                fatigue = st.slider("Fatigue Level (1-3)", 1, 3, 2)
                df_raw = pd.DataFrame([{"BodyTemp": temp, "Chills": chills, "BodyAches": body_aches, "FatigueLevel": fatigue, "Headache": 1, "Cough": 1, "DurationDays": 3}])
            
            elif d_key == "dengue":
                fever = st.number_input("High Fever (°C)", 37.0, 41.5, 39.2, step=0.1)
                retro = st.selectbox("Retro-Orbital Eye Pain?", [0, 1])
                joint_pain = st.selectbox("Severe Joint/Bone Pain?", [0, 1])
                platelets = st.number_input("Platelet Count (k/µL)", 10, 400, 75)
                df_raw = pd.DataFrame([{"HighFever": fever, "RetroOrbitalPain": retro, "SevereJointPain": joint_pain, "PlateletCount": platelets, "PetechiaeRash": 1, "Hematocrit": 46.0}])

            elif d_key == "sepsis":
                hr = st.number_input("Heart Rate (bpm)", 40, 200, 110)
                sys_bp = st.number_input("Systolic BP (mmHg)", 50, 220, 92)
                temp = st.number_input("Body Temp (°C)", 34.0, 42.0, 38.9, step=0.1)
                wbc = st.number_input("WBC Count (k/µL)", 1.0, 50.0, 17.0, step=0.5)
                lactate = st.number_input("Lactate (mmol/L)", 0.5, 20.0, 4.5, step=0.2)
                df_raw = pd.DataFrame([{"HeartRate": hr, "SysBP": sys_bp, "RespRate": 26, "BodyTemp": temp, "WBC_Count": wbc, "Lactate": lactate, "Platelets": 130, "Bilirubin": 1.2, "Creatinine": 2.0}])

            else:
                age = st.number_input("Age", 18, 90, 52)
                bp = st.number_input("Blood Pressure (mmHg)", 80, 220, 135)
                glucose = st.number_input("Glucose (mg/dL)", 60, 400, 130)
                df_raw = pd.DataFrame([{"Age": age, "Sex": 1, "ChestPainType": 3, "RestingBP": bp, "Cholesterol": 220, "FastingBS": 0, "RestingECG": 1, "MaxHR": 145, "ExerciseAngina": 0, "Oldpeak": 1.0, "ST_Slope": 2, "ca": 0, "thal": 2}])

            submitted = st.form_submit_button(f"Compute {cfg['name']} Prediction", use_container_width=True)

        if submitted:
            result = predict_disease_risk(d_key, df_raw)
            if result:
                st.markdown("---")
                c_gauge, c_info = st.columns([1, 1])

                with c_gauge:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=result["percentage"],
                            title={"text": f"{cfg['name']} Risk (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": cfg["color"]},
                                "steps": [
                                    {"range": [0, 35], "color": "rgba(5, 150, 105, 0.2)"},
                                    {"range": [35, 65], "color": "rgba(217, 119, 6, 0.2)"},
                                    {"range": [65, 100], "color": "rgba(220, 38, 38, 0.2)"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(template="plotly_dark", height=280)
                    st.plotly_chart(fig, use_container_width=True)

                with c_info:
                    st.markdown("### Assessment Outcome")
                    badge_cls = result["badge_cls"]
                    category_str = result["category"]
                    percentage_val = result["percentage"]
                    st.markdown(f"<div class='{badge_cls}'>{category_str} ({percentage_val:.1f}%)</div>", unsafe_allow_html=True)
                    st.write("")
                    st.info(f"Model deployed: **{result['metadata']['best_model_name']}** (Test ROC-AUC: {result['metadata']['test_roc_auc']:.3f})")

                if explainer is not None and result["transformed_X"] is not None:
                    st.markdown("---")
                    st.subheader("🔬 SHAP Individual Feature Attribution")
                    try:
                        shap_vals = explainer(result["transformed_X"])
                        fig_shap, ax = plt.subplots(figsize=(8, 4))
                        shap.plots.waterfall(shap_vals[0], show=False)
                        st.pyplot(fig_shap)
                    except Exception as e:
                        st.warning(f"SHAP chart display notice: {e}")


# ---------------------------------------------------------------
# PAGE: MODEL ANALYTICS & BENCHMARKS
# ---------------------------------------------------------------
elif page == "📊 Model Analytics & Benchmarks":
    st.title("📊 20-Disease Model Benchmarks & Comparative Analytics")
    summary_df = load_summary_analytics()

    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)

        fig = px.bar(
            summary_df,
            x="disease",
            y=["test_accuracy", "test_f1", "test_roc_auc"],
            barmode="group",
            title="20-Disease Performance Metrics Comparison (Accuracy, F1-Score, ROC-AUC)",
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python train_model.py` to generate multi-disease benchmark analytics.")


# ---------------------------------------------------------------
# PAGE: ARCHITECTURE & DEFENSE
# ---------------------------------------------------------------
elif page == "ℹ️ Architecture & Defense":
    st.title("ℹ️ Architecture & Zero Data Leakage Defense")
    st.markdown(
        """
        ### 🛡️ Enterprise Guarantees
        1. **Zero Data Leakage**: Feature imputations, scaling, and encodings are isolated inside Scikit-Learn ColumnTransformer pipelines fitted strictly on training folds.
        2. **20-Disease Universal Diagnostic Suite**: Covers common everyday ailments (Fever, Flu, Dengue, Malaria, Typhoid, Cold, Gastroenteritis) alongside chronic, severe, and emergency critical conditions.
        3. **🤖 Ada Interactive AI Health Assessment**: Guided 4-step medical consultation wizard with targeted follow-up questioning and triage badges.
        4. **Explainable AI**: SHAP explainers integrated for all 20 trained models.
        """
    )
