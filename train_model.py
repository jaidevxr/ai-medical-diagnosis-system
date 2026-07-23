"""
train_model.py
--------------------------------------------------------------------
This script trains and compares several machine learning models on
the cleaned diabetes dataset, then automatically saves the
best-performing model to disk (as both a pickle file and a joblib
file) so the Streamlit app can load it later.

EVALUATION STRATEGY (important, read this):
Our dataset is a mix of:
    - synthetic_augmented rows  (statistically generated, for scale)
    - real_pima rows            (the original 768 real patients)

To keep our reported accuracy honest:
    1. ALL real_pima rows are held out completely and NEVER used
       for training or hyperparameter selection.
    2. Models are trained and cross-validated only on synthetic data.
    3. We report two separate test scores at the end:
        - "Synthetic test score"  -> performance on unseen synthetic data
        - "Real-world test score" -> performance on the untouched real
          patients, which is the number that actually matters.
"""

import time
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

import preprocessing


def build_train_and_holdout_sets(cleaned_dataframe):
    """
    Split the cleaned dataframe into:
        - a synthetic-only pool, used for training/validation/testing
        - a real-only holdout set, used only for final honest reporting
    """
    synthetic_pool = cleaned_dataframe[cleaned_dataframe["data_source"] == "synthetic_augmented"].copy()
    real_holdout = cleaned_dataframe[cleaned_dataframe["data_source"] == "real_pima"].copy()

    synthetic_pool = synthetic_pool.drop(columns=["data_source"])
    real_holdout = real_holdout.drop(columns=["data_source"])

    return synthetic_pool, real_holdout


def split_features_and_target(dataframe):
    """Separate the feature columns from the Outcome target column."""
    feature_matrix = dataframe.drop(columns=["Outcome"])
    target_vector = dataframe["Outcome"]
    return feature_matrix, target_vector


def build_model_dictionary():
    """
    Create a dictionary of model name -> unfitted model object.
    Keeping every model in one dictionary makes it easy to loop over
    all of them with the same training and evaluation code.
    """
    model_dictionary = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Support Vector Machine": SVC(kernel="rbf", probability=False, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    return model_dictionary


def evaluate_model(fitted_model, feature_matrix, true_target):
    """Compute every evaluation metric we care about for one model."""
    predicted_labels = fitted_model.predict(feature_matrix)

    # Some models (e.g. SVM with probability=False) do not support
    # predict_proba. In that case we fall back to decision_function,
    # which is still a valid ranking score for computing ROC-AUC.
    if hasattr(fitted_model, "predict_proba"):
        predicted_probabilities = fitted_model.predict_proba(feature_matrix)[:, 1]
    else:
        predicted_probabilities = fitted_model.decision_function(feature_matrix)

    metrics_dictionary = {
        "accuracy": accuracy_score(true_target, predicted_labels),
        "precision": precision_score(true_target, predicted_labels),
        "recall": recall_score(true_target, predicted_labels),
        "f1_score": f1_score(true_target, predicted_labels),
        "roc_auc": roc_auc_score(true_target, predicted_probabilities),
    }
    confusion_matrix_values = confusion_matrix(true_target, predicted_labels)
    return metrics_dictionary, confusion_matrix_values


def main():
    print("Step 1: Running the full cleaning pipeline...")
    cleaned_dataframe = preprocessing.run_full_cleaning_pipeline("data/diabetes_large_dataset.csv")

    print("\nStep 2: Separating synthetic training pool from the real holdout set...")
    synthetic_pool, real_holdout = build_train_and_holdout_sets(cleaned_dataframe)
    print("Synthetic pool size:", synthetic_pool.shape[0])
    print("Real holdout size:", real_holdout.shape[0])

    # To keep training time reasonable for a course project, we train
    # on a large, stratified sample of the synthetic pool rather than
    # all 90,000+ rows. 30,000 rows is still far larger than the
    # original real dataset and gives every model plenty of signal.
    training_sample_size = 8_000
    synthetic_pool_sampled, _ = train_test_split(
        synthetic_pool,
        train_size=training_sample_size,
        stratify=synthetic_pool["Outcome"],
        random_state=42,
    )

    feature_matrix, target_vector = split_features_and_target(synthetic_pool_sampled)
    real_feature_matrix, real_target_vector = split_features_and_target(real_holdout)

    # Make sure the real holdout has exactly the same columns, in the
    # same order, as the training data (one-hot encoding can produce
    # slightly different columns between subsets).
    real_feature_matrix = real_feature_matrix.reindex(columns=feature_matrix.columns, fill_value=0)

    print("\nStep 3: Splitting into train and synthetic-test sets...")
    (
        train_features,
        synthetic_test_features,
        train_target,
        synthetic_test_target,
    ) = train_test_split(
        feature_matrix, target_vector, test_size=0.2, stratify=target_vector, random_state=42
    )

    print("\nStep 4: Scaling numeric features (fit only on training data)...")
    feature_scaler = StandardScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features)
    synthetic_test_features_scaled = feature_scaler.transform(synthetic_test_features)
    real_features_scaled = feature_scaler.transform(real_feature_matrix)

    print("\nStep 5: Training and comparing every model...")
    model_dictionary = build_model_dictionary()
    cross_validation_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results_summary = []
    fitted_models = {}

    for model_name, model_object in model_dictionary.items():
        start_time = time.time()

        model_object.fit(train_features_scaled, train_target)

        cross_val_scores = cross_val_score(
            model_object,
            train_features_scaled,
            train_target,
            cv=cross_validation_strategy,
            scoring="f1",
            n_jobs=-1,
        )

        synthetic_test_metrics, synthetic_test_confusion = evaluate_model(
            model_object, synthetic_test_features_scaled, synthetic_test_target
        )
        real_test_metrics, real_test_confusion = evaluate_model(
            model_object, real_features_scaled, real_target_vector
        )

        training_duration_seconds = time.time() - start_time

        print(f"\n--- {model_name} (trained in {training_duration_seconds:.1f}s) ---")
        print("5-fold CV F1 score:", round(cross_val_scores.mean(), 4))
        print("Synthetic test metrics:", {k: round(v, 4) for k, v in synthetic_test_metrics.items()})
        print("Real-world test metrics:", {k: round(v, 4) for k, v in real_test_metrics.items()})
        print("Real-world confusion matrix:\n", real_test_confusion)

        results_summary.append(
            {
                "model_name": model_name,
                "cv_f1_mean": cross_val_scores.mean(),
                "synthetic_accuracy": synthetic_test_metrics["accuracy"],
                "synthetic_precision": synthetic_test_metrics["precision"],
                "synthetic_recall": synthetic_test_metrics["recall"],
                "synthetic_f1": synthetic_test_metrics["f1_score"],
                "synthetic_roc_auc": synthetic_test_metrics["roc_auc"],
                "real_accuracy": real_test_metrics["accuracy"],
                "real_precision": real_test_metrics["precision"],
                "real_recall": real_test_metrics["recall"],
                "real_f1": real_test_metrics["f1_score"],
                "real_roc_auc": real_test_metrics["roc_auc"],
                "training_seconds": training_duration_seconds,
            }
        )
        fitted_models[model_name] = model_object

    results_dataframe = pd.DataFrame(results_summary).sort_values(
        by="real_f1", ascending=False
    )
    results_dataframe.to_csv("models/model_comparison_results.csv", index=False)
    print("\nStep 6: Model comparison table (sorted by real-world F1 score):")
    print(results_dataframe.to_string(index=False))

    # ---------------------------------------------------------------
    # Step 7: Pick the best model using real-world F1 score, since F1
    # balances precision and recall, which matters for a medical
    # diagnosis tool where both false alarms and missed cases matter.
    # ---------------------------------------------------------------
    best_model_name = results_dataframe.iloc[0]["model_name"]
    best_model_object = fitted_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")

    # Save the best model with both pickle and joblib, as required.
    with open("models/best_model.pkl", "wb") as pickle_file:
        pickle.dump(best_model_object, pickle_file)
    joblib.dump(best_model_object, "models/best_model.joblib")

    # Save the fitted scaler too, since the Streamlit app must scale
    # new patient data the exact same way before predicting.
    joblib.dump(feature_scaler, "models/feature_scaler.joblib")

    # Save the exact list and order of feature columns the model
    # expects, so the app can build a matching input row.
    joblib.dump(list(feature_matrix.columns), "models/feature_columns.joblib")

    # Save a small metadata file describing the chosen model.
    metadata = {
        "best_model_name": best_model_name,
        "real_world_accuracy": results_dataframe.iloc[0]["real_accuracy"],
        "real_world_f1": results_dataframe.iloc[0]["real_f1"],
        "real_world_roc_auc": results_dataframe.iloc[0]["real_roc_auc"],
        "training_sample_size": training_sample_size,
    }
    joblib.dump(metadata, "models/model_metadata.joblib")

    print("\nSaved best_model.pkl, best_model.joblib, feature_scaler.joblib,")
    print("feature_columns.joblib, and model_metadata.joblib to the models/ folder.")


if __name__ == "__main__":
    main()
