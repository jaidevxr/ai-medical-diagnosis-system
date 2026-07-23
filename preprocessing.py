"""
preprocessing.py
--------------------------------------------------------------------
This module holds every data-cleaning and feature-engineering
function used by the project. Keeping these functions in one file
means both the EDA notebook and the training script can reuse the
exact same logic, so the data seen during exploration is guaranteed
to match the data seen during training.

Dataset columns (Pima Indians Diabetes Dataset + synthetic expansion):
    Pregnancies                : number of pregnancies
    Glucose                    : plasma glucose concentration
    BloodPressure              : diastolic blood pressure (mm Hg)
    SkinThickness               : triceps skin fold thickness (mm)
    Insulin                     : 2-hour serum insulin (mu U/ml)
    BMI                         : body mass index
    DiabetesPedigreeFunction    : genetic diabetes risk score
    Age                         : age in years
    Outcome                     : 1 = diabetic, 0 = non-diabetic (target)
    data_source                 : "real_pima" or "synthetic_augmented"
"""

import numpy as np
import pandas as pd

# Columns where a recorded value of 0 is not medically possible.
# In the raw dataset these zeros are actually missing values that
# were encoded as 0 instead of being left blank.
COLUMNS_WHERE_ZERO_MEANS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_raw_dataset(file_path):
    """Load the combined large dataset from disk."""
    raw_dataframe = pd.read_csv(file_path)
    return raw_dataframe


def replace_invalid_zeros_with_missing(input_dataframe):
    """
    Replace medically impossible zero values with NaN so they can be
    handled properly by the missing-value imputation step, instead of
    silently dragging averages down.
    """
    cleaned_dataframe = input_dataframe.copy()
    for column_name in COLUMNS_WHERE_ZERO_MEANS_MISSING:
        zero_mask = cleaned_dataframe[column_name] == 0
        cleaned_dataframe.loc[zero_mask, column_name] = np.nan
    return cleaned_dataframe


def remove_duplicate_rows(input_dataframe):
    """Remove exact duplicate rows from the dataset."""
    number_of_rows_before = input_dataframe.shape[0]
    deduplicated_dataframe = input_dataframe.drop_duplicates()
    number_of_rows_after = deduplicated_dataframe.shape[0]
    number_of_duplicates_removed = number_of_rows_before - number_of_rows_after
    print("Duplicate rows removed:", number_of_duplicates_removed)
    return deduplicated_dataframe


def fill_missing_values_with_median(input_dataframe):
    """
    Fill missing values in each numeric column using the median of
    that column, split by Outcome group. Using the median (rather
    than the mean) makes the fill less sensitive to outliers, and
    splitting by Outcome keeps the imputed values realistic for each
    class rather than blending diabetic and non-diabetic patterns.
    """
    filled_dataframe = input_dataframe.copy()
    for column_name in COLUMNS_WHERE_ZERO_MEANS_MISSING:
        median_by_outcome = filled_dataframe.groupby("Outcome")[column_name].transform("median")
        missing_mask = filled_dataframe[column_name].isna()
        filled_dataframe.loc[missing_mask, column_name] = median_by_outcome[missing_mask]
    return filled_dataframe


def remove_outliers_using_iqr(input_dataframe, columns_to_check):
    """
    Remove rows containing extreme outliers using the Interquartile
    Range (IQR) method. A value is treated as an outlier if it falls
    further than 1.5 times the IQR below the first quartile or above
    the third quartile.
    """
    cleaned_dataframe = input_dataframe.copy()
    for column_name in columns_to_check:
        first_quartile = cleaned_dataframe[column_name].quantile(0.25)
        third_quartile = cleaned_dataframe[column_name].quantile(0.75)
        interquartile_range = third_quartile - first_quartile

        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range

        within_bounds_mask = cleaned_dataframe[column_name].between(lower_bound, upper_bound)
        cleaned_dataframe = cleaned_dataframe[within_bounds_mask]

    cleaned_dataframe = cleaned_dataframe.reset_index(drop=True)
    return cleaned_dataframe


def engineer_new_features(input_dataframe):
    """
    Create additional clinically-meaningful features from the raw
    columns. These derived features often help simple models pick up
    patterns that would otherwise need more complex interactions.
    """
    engineered_dataframe = input_dataframe.copy()

    # BMI category, following standard WHO BMI bands.
    bmi_bin_edges = [0, 18.5, 24.9, 29.9, 100]
    bmi_bin_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    engineered_dataframe["BMI_Category"] = pd.cut(
        engineered_dataframe["BMI"], bins=bmi_bin_edges, labels=bmi_bin_labels
    )

    # Age group, useful because diabetes risk rises with age.
    age_bin_edges = [0, 30, 45, 60, 120]
    age_bin_labels = ["Young", "Middle_Aged", "Senior", "Elderly"]
    engineered_dataframe["Age_Group"] = pd.cut(
        engineered_dataframe["Age"], bins=age_bin_edges, labels=age_bin_labels
    )

    # Glucose category, based on standard clinical glucose thresholds.
    glucose_bin_edges = [0, 100, 125, 500]
    glucose_bin_labels = ["Normal", "Prediabetic", "Diabetic_Range"]
    engineered_dataframe["Glucose_Category"] = pd.cut(
        engineered_dataframe["Glucose"], bins=glucose_bin_edges, labels=glucose_bin_labels
    )

    # A simple interaction feature: glucose per unit of BMI.
    engineered_dataframe["Glucose_to_BMI_Ratio"] = (
        engineered_dataframe["Glucose"] / engineered_dataframe["BMI"]
    )

    # A simple interaction feature combining age and pregnancies,
    # since older patients with more pregnancies show higher risk.
    engineered_dataframe["Age_Pregnancies_Interaction"] = (
        engineered_dataframe["Age"] * engineered_dataframe["Pregnancies"]
    )

    return engineered_dataframe


def one_hot_encode_categorical_features(input_dataframe, categorical_columns):
    """Convert categorical columns into one-hot encoded numeric columns."""
    encoded_dataframe = pd.get_dummies(
        input_dataframe, columns=categorical_columns, drop_first=True
    )
    return encoded_dataframe


def run_full_cleaning_pipeline(file_path):
    """
    Run every cleaning step in order and return a fully cleaned,
    feature-engineered dataframe ready for modeling.
    """
    raw_dataframe = load_raw_dataset(file_path)

    dataframe_with_missing_marked = replace_invalid_zeros_with_missing(raw_dataframe)
    dataframe_without_duplicates = remove_duplicate_rows(dataframe_with_missing_marked)
    dataframe_without_missing = fill_missing_values_with_median(dataframe_without_duplicates)

    columns_to_check_for_outliers = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    dataframe_without_outliers = remove_outliers_using_iqr(
        dataframe_without_missing, columns_to_check_for_outliers
    )

    engineered_dataframe = engineer_new_features(dataframe_without_outliers)

    categorical_columns = ["BMI_Category", "Age_Group", "Glucose_Category"]
    final_dataframe = one_hot_encode_categorical_features(engineered_dataframe, categorical_columns)

    print("Final cleaned dataset shape:", final_dataframe.shape)
    return final_dataframe


if __name__ == "__main__":
    cleaned_data = run_full_cleaning_pipeline("data/diabetes_large_dataset.csv")
    cleaned_data.to_csv("data/diabetes_cleaned.csv", index=False)
    print("Cleaned dataset saved to data/diabetes_cleaned.csv")
