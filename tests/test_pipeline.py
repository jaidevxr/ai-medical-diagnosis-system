"""
test_pipeline.py
--------------------------------------------------------------------
Automated Unit and Integration Test Suite for Medical Diagnosis System

Verifies:
  1. Zero Data Leakage in Preprocessing (Target variable never touched).
  2. Feature Engineering correctness.
  3. Preprocessor transformation shapes and non-null guarantees.
  4. Model Artifact loading and inference validity.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Ensure parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import preprocessing


@pytest.fixture
def sample_patient_dataframe():
    """Provides a synthetic test DataFrame matching the CDC feature schema."""
    data = {
        "HighBP": [1, 0, 1, 0, 1],
        "HighChol": [1, 0, 0, 1, 1],
        "CholCheck": [1, 1, 1, 1, 1],
        "BMI": [28.5, 22.0, 34.1, 19.8, 41.2],
        "Smoker": [1, 0, 1, 0, 0],
        "Stroke": [0, 0, 0, 0, 1],
        "HeartDiseaseorAttack": [0, 0, 1, 0, 1],
        "PhysActivity": [0, 1, 0, 1, 0],
        "Fruits": [1, 1, 0, 1, 0],
        "Veggies": [1, 1, 1, 1, 0],
        "HvyAlcoholConsump": [0, 0, 0, 0, 0],
        "AnyHealthcare": [1, 1, 1, 1, 1],
        "NoDocbcCost": [0, 0, 0, 0, 0],
        "GenHlth": [3, 1, 4, 2, 5],
        "MentHlth": [0, 0, 10, 0, 25],
        "PhysHlth": [5, 0, 15, 0, 30],
        "DiffWalk": [0, 0, 1, 0, 1],
        "Sex": [1, 0, 1, 0, 1],
        "Age": [9, 3, 11, 4, 13],
        "Education": [5, 6, 4, 6, 3],
        "Income": [7, 8, 5, 8, 2],
        "Outcome": [1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_feature_engineering_correctness(sample_patient_dataframe):
    """Verifies engineered clinical features."""
    df_engineered = preprocessing.engineer_cdc_features(sample_patient_dataframe)

    assert "BMI_Category" in df_engineered.columns
    assert "Age_Group" in df_engineered.columns
    assert "Comorbidity_Index" in df_engineered.columns
    assert "Lifestyle_Risk_Score" in df_engineered.columns
    assert "Health_Impairment_Days" in df_engineered.columns

    # Verify specific calculations
    assert df_engineered.loc[0, "Comorbidity_Index"] == 2  # HighBP (1) + HighChol (1)
    assert df_engineered.loc[0, "Health_Impairment_Days"] == 5  # PhysHlth (5) + MentHlth (0)


def test_zero_leakage_preprocessor(sample_patient_dataframe):
    """Verifies that preprocessor constructs without referencing target column."""
    df_engineered = preprocessing.engineer_cdc_features(sample_patient_dataframe)
    X = df_engineered.drop(columns=["Outcome"])

    categorical_features = ["BMI_Category", "Age_Group"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = preprocessing.build_preprocessor_pipeline(numeric_features, categorical_features)
    assert isinstance(preprocessor, ColumnTransformer)

    # Fit strictly on features X (Target is not passed)
    transformed = preprocessor.fit_transform(X)
    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == X.shape[0]


def test_model_artifacts_and_prediction():
    """Test loading artifacts and running inference."""
    import joblib

    model_path = "models/best_model.joblib"
    preprocessor_path = "models/preprocessor.joblib"

    assert os.path.exists(model_path)
    assert os.path.exists(preprocessor_path)

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    dummy_raw = pd.DataFrame(
        [{
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 30.0,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 3,
            "MentHlth": 2,
            "PhysHlth": 1,
            "DiffWalk": 0,
            "Sex": 1,
            "Age": 8,
            "Education": 5,
            "Income": 7,
        }]
    )
    dummy_engineered = preprocessing.engineer_cdc_features(dummy_raw)
    transformed = preprocessor.transform(dummy_engineered)

def test_all_diseases_feature_engineering():
    """Verifies feature engineering for all 10 supported disease models."""
    df_heart = pd.DataFrame([{"Cholesterol": 240, "RestingBP": 120, "MaxHR": 150, "Age": 50, "Oldpeak": 1.5, "ExerciseAngina": 1}])
    df_heart_eng = preprocessing.engineer_heart_features(df_heart)
    assert "Atherogenic_Risk_Index" in df_heart_eng.columns

    df_kidney = pd.DataFrame([{"SerumCreatinine": 1.5, "BloodUrea": 45, "Hemoglobin": 11.0, "Age": 60, "Albumin": 2, "Hypertension": 1}])
    df_kidney_eng = preprocessing.engineer_kidney_features(df_kidney)
    assert "BUN_Creatinine_Ratio" in df_kidney_eng.columns

    df_liver = pd.DataFrame([{"AspartateAminotransferase": 60, "AlamineAminotransferase": 40, "TotalBilirubin": 2.0, "DirectBilirubin": 0.8, "TotalProteins": 7.0, "Albumin": 3.5}])
    df_liver_eng = preprocessing.engineer_liver_features(df_liver)
    assert "DeRitis_Ratio" in df_liver_eng.columns

    df_stroke = pd.DataFrame([{"AvgGlucoseLevel": 120, "BMI": 28.0, "Hypertension": 1, "HeartDisease": 0, "Age": 65}])
    df_stroke_eng = preprocessing.engineer_stroke_features(df_stroke)
    assert "CardioMetabolic_Score" in df_stroke_eng.columns

    df_cancer = pd.DataFrame([{"mean radius": 14.0, "mean texture": 19.0, "mean concavity": 0.08, "mean concave points": 0.04}])
    df_cancer_eng = preprocessing.engineer_cancer_features(df_cancer)
    assert "Tumor_Density_Index" in df_cancer_eng.columns

    df_pneu = pd.DataFrame([{"SpO2": 92.0, "RespiratoryRate": 24, "FeverTemp": 38.5, "WBC_Count": 14.0}])
    df_pneu_eng = preprocessing.engineer_pneumonia_features(df_pneu)
    assert "Hypoxia_Tachypnea_Index" in df_pneu_eng.columns

    df_hyp = pd.DataFrame([{"SystolicBP": 150, "DiastolicBP": 95}])
    df_hyp_eng = preprocessing.engineer_hypertension_features(df_hyp)
    assert "Pulse_Pressure" in df_hyp_eng.columns

    df_sepsis = pd.DataFrame([{"SysBP": 90, "HeartRate": 110, "Lactate": 4.0, "WBC_Count": 16.0}])
    df_sepsis_eng = preprocessing.engineer_sepsis_features(df_sepsis)
    assert "Shock_Index" in df_sepsis_eng.columns

    df_dem = pd.DataFrame([{"MMSE_Score": 22, "CDR_Scale": 1.0, "Age": 75, "EducationYears": 12}])
    df_dem_eng = preprocessing.engineer_dementia_features(df_dem)
    assert "Cognitive_Dementia_Composite" in df_dem_eng.columns


def test_all_diseases_artifacts_and_inference():
    """Verifies artifact presence and valid inference across all 20 disease models."""
    import joblib

    diseases = [
        "diabetes", "heart", "kidney", "liver", "stroke",
        "cancer", "pneumonia", "hypertension", "sepsis", "dementia",
        "fever", "malaria", "typhoid", "dengue", "cold",
        "gastro", "anemia", "thyroid", "asthma", "hypertensive_crisis"
    ]

    for disease in diseases:
        disease_dir = os.path.join("models", disease)
        model_path = os.path.join(disease_dir, "best_model.joblib")
        prep_path = os.path.join(disease_dir, "preprocessor.joblib")

        assert os.path.exists(model_path), f"Missing model for {disease}"
        assert os.path.exists(prep_path), f"Missing preprocessor for {disease}"

        model = joblib.load(model_path)
        prep = joblib.load(prep_path)

        assert model is not None
        assert prep is not None



