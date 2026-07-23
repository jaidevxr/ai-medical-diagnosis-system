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

    pred = model.predict(transformed)
    assert pred[0] in [0, 1]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(transformed)[0, 1]
        assert 0.0 <= prob <= 1.0
