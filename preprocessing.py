"""
preprocessing.py
--------------------------------------------------------------------
Zero-Leakage Data Processing & Feature Engineering Module

Guarantees 0% Data Leakage:
  1. ALL feature imputation and scaling operations are encapsulated
     inside Scikit-Learn Pipelines and ColumnTransformers.
  2. Transformers are fitted STRICTLY on training data (`fit_transform`)
     and applied (`transform`) to validation/test sets.
  3. Target variable `Outcome` is NEVER used during feature imputation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_dataset(file_path):
    """Load clinical dataset from disk."""
    return pd.read_csv(file_path)


def engineer_cdc_features(df):
    """
    Applies medically meaningful domain feature engineering to the CDC
    Diabetes Health Indicators dataset.
    
    Features engineered:
      - BMI_Category: WHO standard body mass index categories.
      - Age_Group: Binned age categories.
      - Comorbidity_Index: Count of co-occurring chronic conditions.
      - Lifestyle_Risk_Score: Unhealthy lifestyle risk factors score.
      - Health_Impairment_Days: Total days of poor physical + mental health.
    """
    df = df.copy()

    # 1. BMI Category (WHO standard)
    if "BMI" in df.columns:
        bmi_edges = [0, 18.5, 24.9, 29.9, 100]
        bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
        df["BMI_Category"] = pd.cut(df["BMI"], bins=bmi_edges, labels=bmi_labels, include_lowest=True)

    # 2. Age Group (CDC Age variable is 1 to 13)
    if "Age" in df.columns:
        # Age 1=18-24, 2=25-29, ..., 13=80+
        age_edges = [0, 3, 7, 10, 14]
        age_labels = ["Young_Adult", "Middle_Aged", "Senior", "Elderly"]
        df["Age_Group"] = pd.cut(df["Age"], bins=age_edges, labels=age_labels, include_lowest=True)

    # 3. Comorbidity Index (Sum of HighBP, HighChol, HeartDiseaseorAttack, Stroke)
    comorbidity_cols = [c for c in ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"] if c in df.columns]
    if comorbidity_cols:
        df["Comorbidity_Index"] = df[comorbidity_cols].sum(axis=1)

    # 4. Lifestyle Risk Score
    lifestyle_risk = pd.Series(0, index=df.index)
    if "Smoker" in df.columns:
        lifestyle_risk += df["Smoker"]
    if "PhysActivity" in df.columns:
        lifestyle_risk += (1 - df["PhysActivity"])
    if "HvyAlcoholConsump" in df.columns:
        lifestyle_risk += df["HvyAlcoholConsump"]
    if "Fruits" in df.columns:
        lifestyle_risk += (1 - df["Fruits"])
    if "Veggies" in df.columns:
        lifestyle_risk += (1 - df["Veggies"])
    df["Lifestyle_Risk_Score"] = lifestyle_risk

    # 5. Total Health Impairment Days
    if "PhysHlth" in df.columns and "MentHlth" in df.columns:
        df["Health_Impairment_Days"] = df["PhysHlth"] + df["MentHlth"]

    return df


def build_preprocessor_pipeline(numeric_features, categorical_features):
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline.
    
    Guarantees:
      - Numerical pipeline: SimpleImputer(median) -> StandardScaler()
      - Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(ignore)
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    return preprocessor


def get_feature_names_after_preprocessing(preprocessor, numeric_features, categorical_features):
    """
    Extracts exact output feature column names from a fitted ColumnTransformer.
    """
    feature_names = []

    # Numerical features remain the same
    feature_names.extend(numeric_features)

    # Categorical features expanded by OneHotEncoder
    if categorical_features and "cat" in preprocessor.named_transformers_:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        onehot_encoder = cat_pipeline.named_steps["onehot"]
        encoded_cat_names = onehot_encoder.get_feature_names_out(categorical_features)
        feature_names.extend(list(encoded_cat_names))

    return feature_names
