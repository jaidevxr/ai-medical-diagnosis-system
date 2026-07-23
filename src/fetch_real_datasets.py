"""
fetch_real_datasets.py
--------------------------------------------------------------------
Fetches and cleans genuine, large-scale clinical datasets for the
Diabetes Diagnosis System.

Primary Dataset:
  - CDC Diabetes Health Indicators Dataset (UCI ID 891 / BRFSS 2015)
  - 70,692 or 253,680 real patient records with 21 clinical, lifestyle,
    and demographic health indicators.
  - Source: CDC BRFSS / UCI Machine Learning Repository

Secondary Dataset:
  - Pima Indians Diabetes Dataset (NIDDK, 768 real patients)
"""

import os
import sys
import pandas as pd
import numpy as np


def fetch_cdc_diabetes_dataset():
    """
    Fetch the CDC Diabetes Health Indicators dataset directly from
    the UCI Machine Learning Repository via `ucimlrepo` package,
    with a fallback direct URL download if needed.
    """
    print("Fetching CDC Diabetes Health Indicators dataset (UCI ID 891)...")
    
    dataframe = None
    try:
        from ucimlrepo import fetch_ucirepo
        cdc_dataset = fetch_ucirepo(id=891)
        features = cdc_dataset.data.features
        targets = cdc_dataset.data.targets
        
        dataframe = pd.concat([features, targets], axis=1)
        print(f"Successfully fetched CDC dataset via ucimlrepo! Shape: {dataframe.shape}")
    except Exception as exc:
        print(f"ucimlrepo fetch encountered issue ({exc}). Trying direct repository fallback...")
        
        fallback_urls = [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "https://archive.ics.uci.edu/static/public/891/cdc+diabetes+health+indicators.zip"
        ]
        
        # If ucimlrepo failed, we can download from direct public mirrors
        import urllib.request
        import zipfile
        import io
        
        try:
            zip_url = "https://archive.ics.uci.edu/static/public/891/cdc+diabetes+health+indicators.zip"
            req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                zip_buffer = io.BytesIO(response.read())
                with zipfile.ZipFile(zip_buffer) as zf:
                    # look for csv inside
                    csv_names = [name for name in zf.namelist() if name.endswith('.csv')]
                    if csv_names:
                        with zf.open(csv_names[0]) as csv_file:
                            dataframe = pd.read_csv(csv_file)
                            print(f"Successfully loaded CDC dataset from UCI ZIP archive! Shape: {dataframe.shape}")
        except Exception as zip_exc:
            print(f"ZIP fetch failed: {zip_exc}. Using local/mirrored dataset fallback.")

    if dataframe is None or dataframe.empty:
        raise RuntimeError("Failed to load CDC Diabetes dataset. Please verify internet connection.")

    # Standardize target column name to Outcome (0 = No Diabetes, 1 = Diabetes)
    target_candidates = ["Diabetes_binary", "Diabetes_01", "Outcome", "target"]
    for col in target_candidates:
        if col in dataframe.columns:
            dataframe = dataframe.rename(columns={col: "Outcome"})
            break
            
    # Remove any NaN or duplicate rows
    initial_count = dataframe.shape[0]
    dataframe = dataframe.dropna().drop_duplicates().reset_index(drop=True)
    final_count = dataframe.shape[0]
    print(f"Cleaned dataset: Removed {initial_count - final_count} duplicates/missing rows. Final shape: {dataframe.shape}")

    # Ensure integer outcome
    dataframe["Outcome"] = dataframe["Outcome"].astype(int)
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/cdc_diabetes_real_large.csv"
    dataframe.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} ({final_count} rows, {dataframe.shape[1]} columns)")
    print("Class balance:\n", dataframe["Outcome"].value_counts(normalize=True))
    
    return dataframe


if __name__ == "__main__":
    fetch_cdc_diabetes_dataset()
