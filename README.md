# 🩺 20-Disease AI Medical Diagnosis & Decision Support System

An enterprise-grade, zero-leakage Machine Learning platform for clinical multi-disease risk prediction, supporting **20 medical condition domains** spanning everyday common ailments to extreme critical emergencies:

### 🟢 Everyday & Common Acute Ailments
1. 🤒 **Fever & Acute Viral Flu**
2. 🦟 **Malaria & Vector-Borne Fever**
3. 🦠 **Typhoid Fever & Enteric Fever**
4. 🩸 **Dengue Fever / Hemorrhagic Risk**
5. 🤧 **Common Cold & Upper Respiratory Infection**
6. 🫄 **Acute Gastroenteritis & Food Poisoning**

### 🟡 Routine & Chronic Conditions
7. 🩸 **Diabetes Mellitus** (CDC BRFSS / Glucose & Metabolic Indicators)
8. 🩸 **Hypertension & Vascular Strain**
9. 🩸 **Anemia & Iron Deficiency**
10. 🦋 **Thyroid Disorder (Hypo/Hyperthyroidism)**
11. ⚡ **Alzheimer's & Dementia Cognitive Impairment**

### 🟠 Severe Organic & Respiratory Diseases
12. ❤️ **Coronary Heart Disease** (UCI Cardiac Biomarkers)
13. 🧪 **Chronic Kidney Disease (CKD)** (Renal Function Panel)
14. 🫀 **Hepatic / Liver Disease** (Indian Liver Patient Dataset)
15. 🫁 **Pneumonia & Respiratory Strain**
16. 🫁 **Asthma & Bronchial Hyperreactivity**
17. 🎗️ **Oncology / Tumor Risk Assessment** (Tissue Biomarkers)

### 🔴 Critical Emergencies
18. 🧠 **Stroke & Cerebrovascular Attack** (Acute Neurological Emergency)
19. 🦠 **Sepsis & Critical Care Shock** (SIRS Criteria & Septic Shock)
20. 💥 **Hypertensive Crisis Risk** (Extreme BP Elevation >180/120)

Built with classic Python data science tools: **NumPy, Pandas, Scikit-Learn, XGBoost, SHAP, Streamlit, Plotly, Seaborn, and Matplotlib**.

---

## 1. Project Architecture

```
Medical-Diagnosis-System/
│
├── data/
│   ├── cdc_diabetes_real_large.csv     # 229,474 real patient records from CDC BRFSS
│   ├── heart_disease_real.csv          # Cardiac clinical dataset
│   ├── kidney_disease_real.csv         # Chronic Kidney Disease clinical dataset
│   ├── liver_disease_real.csv          # Indian Liver Patient clinical dataset
│   ├── stroke_prediction_real.csv      # Healthcare Stroke Prediction dataset
│   ├── fever_viral_real.csv            # Fever & Flu clinical dataset
│   ├── malaria_fever_real.csv          # Malaria clinical dataset
│   ├── dengue_fever_real.csv           # Dengue clinical dataset
│   └── ... (20 total disease clinical datasets)
│
├── models/
│   ├── diabetes/                       # Diabetes ML model & SHAP artifacts
│   ├── heart/                          # Heart Disease ML model & SHAP artifacts
│   ├── kidney/                         # Kidney Disease ML model & SHAP artifacts
│   ├── ...                             # (20 disease model artifact directories)
│   └── all_diseases_summary.csv        # Benchmark comparison table for all 20 diseases
│
├── src/
│   ├── fetch_real_datasets.py          # Fetches & prepares datasets for all 20 diseases
│   └── build_eda_notebook.py           # Programmatically generates EDA notebook
│
├── tests/
│   └── test_pipeline.py                # Pytest suite verifying all 20 disease models
│
├── app.py                              # Streamlit app with Universal 20-Disease Scanner
├── preprocessing.py                    # Multi-Disease Zero-Leakage Pipeline & Feature Engineering
├── train_model.py                      # Multi-disease model training & SHAP explainer generator
├── requirements.txt
└── README.md
```

---

## 2. Dataset & Zero Data Leakage Guarantee

### **Real Dataset (CDC BRFSS / UCI ID 891)**
- **70,692 balanced patient records** with 21 clinical, lifestyle, and demographic health indicators:
  - Clinical: `BMI`, `HighBP`, `HighChol`, `CholCheck`, `Stroke`, `HeartDiseaseorAttack`
  - Lifestyle: `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`
  - Health Status: `GenHlth`, `PhysHlth`, `MentHlth`, `DiffWalk`
  - Demographics: `Age`, `Sex`, `Education`, `Income`

### **Zero Data Leakage Guarantee**
1. **Target Isolation**: Target variable `Outcome` is **never** used during imputation or feature scaling.
2. **Pipeline Encapsulation**: All preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) is strictly encapsulated inside Scikit-Learn `Pipeline` and `ColumnTransformer` objects.
3. **Strict Fold Isolation**: Transformers are fitted **only** on training data (`X_train`) inside cross-validation folds and transformed onto test sets (`X_test`).

---

## 3. Quickstart & Execution Guide

```bash
# 1. Clone the repository
git clone https://github.com/jaidevxr/ai-medical-diagnosis-system.git
cd ai-medical-diagnosis-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch real CDC clinical dataset
python src/fetch_real_datasets.py

# 4. Train, compare models, and fit SHAP explainers
python train_model.py

# 5. Run unit tests
pytest tests/

# 6. Launch interactive Streamlit application
streamlit run app.py
```

---

## 4. Model Suite & Benchmark Results

Eight classifiers are evaluated under **5-Fold Stratified Cross-Validation** and tested on unseen test data:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Naive Bayes
7. Gradient Boosting
8. XGBoost

Metrics computed per model: **Accuracy, Precision, Recall, F1 Score, ROC-AUC, PR-AUC, Confusion Matrix, and CV F1 Mean/Std**.

---

## 5. SHAP Model Explainability

Every patient prediction on the **Disease Prediction** page is accompanied by an individual **SHAP Waterfall Plot**, displaying:
- Top positive risk factors increasing diabetes probability.
- Top negative protective factors lowering diabetes probability.
- Baseline population probability vs. individual patient risk.

---

## 6. Streamlit Application Pages

| Page | Description |
|---|---|
| **Home** | Project overview, zero-leakage guarantee, headline model metrics |
| **Disease Prediction** | Patient input form → real-time prediction, probability gauge, risk level, SHAP explanation |
| **SHAP Explainability** | Global feature importance rankings and population SHAP analysis |
| **Dataset Insights** | CDC dataset preview, statistics, pie charts, distributions, correlation heatmap |
| **Model Performance** | Head-to-head comparison table and interactive Plotly bar charts |
| **Graph Dashboard** | Interactive box plots, scatter plots, multi-variable relationships |
| **About** | System architecture, tech stack, data sources, clinical disclaimer |

---

## 7. Disclaimer

This system is an **educational and clinical decision-support research tool**. It is **not** a certified medical device and must never replace professional medical advice, diagnosis, or treatment.
