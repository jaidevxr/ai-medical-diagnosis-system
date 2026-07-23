# 🩺 AI-Powered Diabetes Diagnosis System

An end-to-end machine learning project that predicts a patient's diabetes
risk from clinical measurements, built entirely with classic (non-deep-learning)
Python data science tools: **NumPy, Pandas, Matplotlib, Seaborn, Plotly,
Scikit-learn, XGBoost, and Streamlit.**

---

## 1. Project Structure

```
Medical-Diagnosis-System/
│
├── data/
│   ├── pima_diabetes_raw.csv          # Real data, exactly as downloaded
│   ├── pima_diabetes_real.csv         # Real data, with column headers added
│   ├── diabetes_large_dataset.csv     # Real + synthetic, combined (100,768 rows)
│   └── diabetes_cleaned.csv           # After the full cleaning pipeline
│
├── notebooks/
│   └── EDA_Diabetes_Analysis.ipynb    # Full exploratory data analysis, fully executed
│
├── models/
│   ├── best_model.pkl                 # Best model (pickle format)
│   ├── best_model.joblib              # Best model (joblib format)
│   ├── feature_scaler.joblib          # Fitted StandardScaler
│   ├── feature_columns.joblib         # Exact column order the model expects
│   ├── model_metadata.joblib          # Which model won, and its real-world scores
│   └── model_comparison_results.csv   # All 8 models, side by side
│
├── src/
│   ├── generate_large_dataset.py      # Builds the 100k-row training dataset
│   └── build_eda_notebook.py          # Programmatically builds/executes the notebook
│
├── app.py                             # Streamlit application (6 pages)
├── preprocessing.py                   # Shared cleaning / feature engineering functions
├── train_model.py                     # Trains, compares, and saves the best model
├── requirements.txt
└── README.md
```

---

## 2. Dataset — Source and Honest Disclosure

**Base dataset (real, public):** the **Pima Indians Diabetes Dataset**,
originally collected by the National Institute of Diabetes and Digestive
and Kidney Diseases (NIDDK). It contains **768 real patient records**
(all female, Pima Indian heritage, age 21+) with 8 clinical features and
a binary `Outcome` label (1 = diabetic, 0 = non-diabetic).

Source URL (fetched automatically by this project):
`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`

**Why there's also a "synthetic" dataset:** the assignment asked for
100,000+ records to demonstrate the pipeline at scale. There is no
genuinely public, freely downloadable diabetes dataset of that size
that could be fetched automatically in this environment (the well-known
100k-row Kaggle `diabetes_prediction_dataset.csv` requires a Kaggle
account and API key to download). Rather than mislabeling a fabricated
file as "real," `src/generate_large_dataset.py`:

1. Loads the real 768-row dataset.
2. Fits a multivariate normal distribution to each outcome class
   (diabetic / non-diabetic) separately, capturing both the spread of
   each feature **and** how features correlate with each other.
3. Draws 100,000 new synthetic rows from those distributions, clips
   them to medically plausible ranges, and tags them
   `data_source = "synthetic_augmented"`.
4. Keeps the original 768 real rows completely separate and untouched,
   tagged `data_source = "real_pima"`.

**Every accuracy number in this project that matters is measured only
on the untouched real patient data** — the model never trains on a
single real row. This keeps the reported performance honest.

---

## 3. Installation Guide

```bash
# 1. Clone or download this project folder
cd Medical-Diagnosis-System

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset (already included, but re-runnable)
python src/generate_large_dataset.py

# 5. Train and save the best model
python train_model.py

# 6. Launch the Streamlit app
streamlit run app.py
```

---

## 4. Data Analysis & Cleaning Pipeline (`preprocessing.py`)

1. **Invalid zero handling** — `Glucose`, `BloodPressure`, `SkinThickness`,
   `Insulin`, and `BMI` cannot medically be 0. These are converted to
   `NaN` before anything else.
2. **Missing value imputation** — filled using the **median per Outcome
   group**, so imputed values stay realistic for diabetic vs.
   non-diabetic patients.
3. **Duplicate removal** — exact duplicate rows are dropped.
4. **Outlier removal** — the IQR method removes statistically extreme
   values in each clinical column.
5. **Feature engineering** — adds `BMI_Category`, `Age_Group`,
   `Glucose_Category` (clinically standard bins), plus two interaction
   features: `Glucose_to_BMI_Ratio` and `Age_Pregnancies_Interaction`.
6. **Encoding** — categorical bins are one-hot encoded.

All of this logic lives in one shared file so the notebook and the
training script can never drift out of sync.

---

## 5. Model Explanation (`train_model.py`)

Eight models are trained and compared:

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (bagging) |
| K-Nearest Neighbors | Distance-based |
| Support Vector Machine | Margin-based |
| Naive Bayes | Probabilistic |
| Gradient Boosting | Ensemble (boosting) |
| XGBoost | Ensemble (boosting) |

Each model is evaluated with **accuracy, precision, recall, F1 score,
ROC-AUC, a confusion matrix, and 5-fold stratified cross-validation**,
computed separately on a **synthetic held-out test set** and the
**untouched real-world test set**.

The model with the best **real-world F1 score** (chosen because F1
balances false alarms and missed diagnoses — both matter in a medical
context) is automatically saved with both `pickle` and `joblib`.

In our run, **XGBoost** was selected, reaching roughly **90% real-world
accuracy** and **0.96 real-world ROC-AUC** on the untouched 705 real
patients not used in training. Exact numbers are in
`models/model_comparison_results.csv` and may vary slightly on re-run.

---

## 6. Streamlit Application (`app.py`)

| Page | Purpose |
|---|---|
| Home | Project overview and headline model metrics |
| Disease Prediction | Real-time form → prediction, confidence score, probability gauge, risk level |
| Dataset Insights | EDA-style charts on the real patient data |
| Model Performance | Side-by-side comparison of all 8 models |
| Graph Dashboard | Interactive feature explorer, box plots, pair plots |
| About | Tech stack, dataset, and disclaimer |

---

## 7. Future Improvements

- Add true real-world data at 100k+ scale once a licensed, larger
  clinical dataset becomes available (e.g. via an institutional data
  use agreement), replacing the synthetic augmentation entirely.
- Extend to additional diseases (e.g. heart disease, chronic kidney
  disease) as separate, equally well-validated prediction modules.
- Add SHAP-based feature explanations to the prediction page so users
  can see *why* a prediction was made.
- Add hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
  for further accuracy gains.
- Add user authentication and a database layer if this were to move
  beyond a portfolio demo toward real deployment.

---

## 8. Disclaimer

This project is an **educational portfolio piece**. It is **not** a
certified medical device and must never be used as a substitute for
professional medical advice, diagnosis, or treatment.
