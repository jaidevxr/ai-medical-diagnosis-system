# 🩺 Enterprise AI-Powered Diabetes Diagnosis & Decision Support System

An enterprise-grade, zero-leakage Machine Learning platform for clinical diabetes risk prediction, trained on **70,692 real CDC patient records** (Behavioral Risk Factor Surveillance System / UCI ML Repository ID 891).

Built with classic Python data science tools: **NumPy, Pandas, Scikit-Learn, XGBoost, SHAP, Streamlit, Plotly, Seaborn, and Matplotlib**.

---

## 1. Project Architecture

```
Medical-Diagnosis-System/
│
├── data/
│   ├── cdc_diabetes_real_large.csv     # 70,692 real patient records from CDC BRFSS / UCI ID 891
│   └── pima_diabetes_real.csv          # 768 real patient records (NIDDK benchmark dataset)
│
├── models/
│   ├── best_model.joblib               # Winning classifier (joblib)
│   ├── best_model.pkl                  # Winning classifier (pickle)
│   ├── preprocessor.joblib             # Fitted Scikit-Learn ColumnTransformer pipeline
│   ├── feature_columns.joblib          # Output feature column names
│   ├── shap_explainer.joblib           # Fitted SHAP TreeExplainer / KernelExplainer
│   ├── model_metadata.joblib           # Best model metrics and dataset metadata
│   └── model_comparison_results.csv    # 8-model benchmark comparison table
│
├── src/
│   ├── fetch_real_datasets.py          # Fetches & cleans real CDC dataset via ucimlrepo / UCI API
│   └── build_eda_notebook.py           # Programmatically generates EDA notebook
│
├── tests/
│   └── test_pipeline.py                # Automated Pytest suite (Zero Leakage, Pipeline, Inference)
│
├── app.py                              # Modern 7-page Streamlit Application with SHAP
├── preprocessing.py                    # Zero-Leakage Pipeline & Feature Engineering
├── train_model.py                      # 8-model suite training, hyperparameter tuning & SHAP
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
