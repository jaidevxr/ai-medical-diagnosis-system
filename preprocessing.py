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


def engineer_heart_features(df):
    """Domain feature engineering for Cardiac/Heart Disease dataset."""
    df = df.copy()
    if "Cholesterol" in df.columns and "RestingBP" in df.columns:
        df["Atherogenic_Risk_Index"] = df["Cholesterol"] / (df["RestingBP"] + 1e-5)
    if "MaxHR" in df.columns and "Age" in df.columns:
        df["HR_Reserve_Ratio"] = df["MaxHR"] / (220 - df["Age"] + 1e-5)
    if "Oldpeak" in df.columns and "ExerciseAngina" in df.columns:
        df["Ischemic_Severity_Score"] = df["Oldpeak"] * (1 + df["ExerciseAngina"])
    return df


def engineer_kidney_features(df):
    """Domain feature engineering for Chronic Kidney Disease (CKD) dataset."""
    df = df.copy()
    if "SerumCreatinine" in df.columns and "BloodUrea" in df.columns:
        df["BUN_Creatinine_Ratio"] = df["BloodUrea"] / (df["SerumCreatinine"] + 1e-5)
    if "Hemoglobin" in df.columns and "Age" in df.columns:
        df["Anemia_Age_Risk"] = (df["Hemoglobin"] < 12.0).astype(int) * (df["Age"] > 50).astype(int)
    if "Albumin" in df.columns and "Hypertension" in df.columns:
        df["Renal_Vascular_Strain"] = df["Albumin"] + 2 * df["Hypertension"]
    return df


def engineer_liver_features(df):
    """Domain feature engineering for Hepatic/Liver Disease dataset."""
    df = df.copy()
    if "AspartateAminotransferase" in df.columns and "AlamineAminotransferase" in df.columns:
        df["DeRitis_Ratio"] = df["AspartateAminotransferase"] / (df["AlamineAminotransferase"] + 1e-5)
    if "TotalBilirubin" in df.columns and "DirectBilirubin" in df.columns:
        df["Indirect_Bilirubin"] = (df["TotalBilirubin"] - df["DirectBilirubin"]).clip(lower=0)
    if "Albumin" in df.columns and "TotalProteins" in df.columns:
        df["Globulin_Level"] = (df["TotalProteins"] - df["Albumin"]).clip(lower=0)
    return df


def engineer_stroke_features(df):
    """Domain feature engineering for Stroke Prediction dataset."""
    df = df.copy()
    if "AvgGlucoseLevel" in df.columns and "BMI" in df.columns:
        df["CardioMetabolic_Score"] = (df["AvgGlucoseLevel"] / 100.0) * (df["BMI"] / 25.0)
    if "Hypertension" in df.columns and "HeartDisease" in df.columns and "Age" in df.columns:
        df["Vascular_Risk_Factor"] = (df["Hypertension"] + df["HeartDisease"]) * (df["Age"] / 50.0)
    return df


def engineer_cancer_features(df):
    """Domain feature engineering for Oncology / Tumor Risk dataset."""
    df = df.copy()
    if "mean radius" in df.columns and "mean texture" in df.columns:
        df["Tumor_Density_Index"] = df["mean radius"] * df["mean texture"]
    if "mean concavity" in df.columns and "mean concave points" in df.columns:
        df["Nuclear_Atypia_Score"] = df["mean concavity"] + 2.5 * df["mean concave points"]
    return df


def engineer_pneumonia_features(df):
    """Domain feature engineering for Pneumonia / Respiratory Disease dataset."""
    df = df.copy()
    if "SpO2" in df.columns and "RespiratoryRate" in df.columns:
        df["Hypoxia_Tachypnea_Index"] = (100.0 - df["SpO2"]) * (df["RespiratoryRate"] / 20.0)
    if "FeverTemp" in df.columns and "WBC_Count" in df.columns:
        df["Systemic_Inflammation_Index"] = (df["FeverTemp"] - 37.0).clip(lower=0) * df["WBC_Count"]
    return df


def engineer_hypertension_features(df):
    """Domain feature engineering for Hypertension & Vascular Strain dataset."""
    df = df.copy()
    if "SystolicBP" in df.columns and "DiastolicBP" in df.columns:
        df["Pulse_Pressure"] = df["SystolicBP"] - df["DiastolicBP"]
        df["Mean_Arterial_Pressure"] = df["DiastolicBP"] + (df["Pulse_Pressure"] / 3.0)
    return df


def engineer_sepsis_features(df):
    """Domain feature engineering for Sepsis & Critical Care Shock dataset."""
    df = df.copy()
    if "SysBP" in df.columns and "HeartRate" in df.columns:
        df["Shock_Index"] = df["HeartRate"] / (df["SysBP"] + 1e-5)
    if "Lactate" in df.columns and "WBC_Count" in df.columns:
        df["Metabolic_Crisis_Score"] = df["Lactate"] * (df["WBC_Count"] / 10.0)
    return df


def engineer_fever_features(df):
    """Domain feature engineering for Fever & Acute Viral Flu dataset."""
    df = df.copy()
    if "BodyTemp" in df.columns and "Chills" in df.columns:
        df["Pyrexia_Index"] = (df["BodyTemp"] - 37.0).clip(lower=0) * (1 + df["Chills"])
    return df


def engineer_malaria_features(df):
    """Domain feature engineering for Malaria & Vector-Borne Fever dataset."""
    df = df.copy()
    if "TempSpike" in df.columns and "PlateletCount" in df.columns:
        df["Thrombocytopenic_Fever_Ratio"] = (df["TempSpike"] - 37.0).clip(lower=0) / (df["PlateletCount"] / 100.0 + 1e-5)
    return df


def engineer_typhoid_features(df):
    """Domain feature engineering for Typhoid Fever dataset."""
    df = df.copy()
    if "StepladderFever" in df.columns and "FeverDuration" in df.columns:
        df["Enteric_Fever_Index"] = df["StepladderFever"] * df["FeverDuration"]
    return df


def engineer_dengue_features(df):
    """Domain feature engineering for Dengue Fever dataset."""
    df = df.copy()
    if "HighFever" in df.columns and "PlateletCount" in df.columns:
        df["Hemorrhagic_Risk_Score"] = (df["HighFever"] - 37.0).clip(lower=0) * (150.0 / (df["PlateletCount"] + 1e-5))
    return df


def engineer_cold_features(df):
    """Domain feature engineering for Common Cold dataset."""
    df = df.copy()
    cols = [c for c in ["Rhinorrhea", "SoreThroat", "Sneezing", "NasalCongestion"] if c in df.columns]
    if cols:
        df["URI_Symptom_Count"] = df[cols].sum(axis=1)
    return df


def engineer_gastro_features(df):
    """Domain feature engineering for Acute Gastroenteritis dataset."""
    df = df.copy()
    if "VomitingEpisodes" in df.columns and "DiarrheaEpisodes" in df.columns:
        df["Fluid_Loss_Severity"] = df["VomitingEpisodes"] + 1.5 * df["DiarrheaEpisodes"]
    return df


def engineer_anemia_features(df):
    """Domain feature engineering for Anemia dataset."""
    df = df.copy()
    if "Hemoglobin" in df.columns and "RBC_Count" in df.columns:
        df["MCH_Proxy"] = df["Hemoglobin"] / (df["RBC_Count"] + 1e-5)
    return df


def engineer_thyroid_features(df):
    """Domain feature engineering for Thyroid Disorder dataset."""
    df = df.copy()
    if "TSH" in df.columns and "Free_T4" in df.columns:
        df["TSH_FT4_Ratio"] = df["TSH"] / (df["Free_T4"] + 1e-5)
    return df


def engineer_asthma_features(df):
    """Domain feature engineering for Asthma dataset."""
    df = df.copy()
    if "PeakExpiratoryFlow" in df.columns and "Wheezing" in df.columns:
        df["Airway_Obstruction_Score"] = (100 - df["PeakExpiratoryFlow"]) * (1 + df["Wheezing"])
    return df


def engineer_hypertensive_crisis_features(df):
    """Domain feature engineering for Hypertensive Crisis dataset."""
    df = df.copy()
    if "SystolicBP" in df.columns and "DiastolicBP" in df.columns:
        df["Extreme_BP_Ratio"] = (df["SystolicBP"] / 140.0) * (df["DiastolicBP"] / 90.0)
    return df


def engineer_dementia_features(df):
    """Domain feature engineering for Alzheimer's & Dementia Cognitive Impairment dataset."""
    df = df.copy()
    if "MMSE_Score" in df.columns and "CDR_Scale" in df.columns:
        df["Cognitive_Dementia_Composite"] = (30 - df["MMSE_Score"]) * (1 + 2 * df["CDR_Scale"])
    if "Age" in df.columns and "EducationYears" in df.columns:
        df["Cognitive_Reserve_Ratio"] = df["EducationYears"] / (df["Age"] / 50.0 + 1e-5)
    return df


def engineer_disease_features(df, disease_name):
    """Engineers clinical features based on target disease key across all 20 disease models."""
    name = disease_name.lower()
    if "diabetes" in name:
        return engineer_cdc_features(df)
    elif "heart" in name or "cardio" in name:
        return engineer_heart_features(df)
    elif "kidney" in name or "ckd" in name:
        return engineer_kidney_features(df)
    elif "liver" in name or "hepatic" in name:
        return engineer_liver_features(df)
    elif "stroke" in name:
        return engineer_stroke_features(df)
    elif "cancer" in name or "oncology" in name or "tumor" in name:
        return engineer_cancer_features(df)
    elif "pneumonia" in name or "respiratory" in name:
        return engineer_pneumonia_features(df)
    elif "hypertension_crisis" in name or "hypertensive_crisis" in name:
        return engineer_hypertensive_crisis_features(df)
    elif "hypertension" in name or "vascular" in name:
        return engineer_hypertension_features(df)
    elif "sepsis" in name or "critical" in name:
        return engineer_sepsis_features(df)
    elif "dementia" in name or "alzheimer" in name or "cognitive" in name:
        return engineer_dementia_features(df)
    elif "fever" in name or "flu" in name:
        return engineer_fever_features(df)
    elif "malaria" in name:
        return engineer_malaria_features(df)
    elif "typhoid" in name:
        return engineer_typhoid_features(df)
    elif "dengue" in name:
        return engineer_dengue_features(df)
    elif "cold" in name:
        return engineer_cold_features(df)
    elif "gastro" in name:
        return engineer_gastro_features(df)
    elif "anemia" in name:
        return engineer_anemia_features(df)
    elif "thyroid" in name:
        return engineer_thyroid_features(df)
    elif "asthma" in name:
        return engineer_asthma_features(df)
    return df.copy()




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

