"""
app.py
--------------------------------------------------------------------
Streamlit application for the AI-Powered Diabetes Diagnosis System.

Pages:
    1. Home                 - project overview
    2. Disease Prediction   - real-time diabetes risk prediction form
    3. Dataset Insights     - EDA-style charts about the real dataset
    4. Model Performance    - comparison of every trained model
    5. Graph Dashboard      - extra interactive Plotly visualizations
    6. About                - project/team/tech-stack information

Run with:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import preprocessing

# ---------------------------------------------------------------
# PAGE CONFIGURATION (must be the first Streamlit command)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Diabetes Diagnosis System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------
# CACHED LOADERS
# Streamlit re-runs the whole script on every user interaction, so
# we cache expensive operations (loading data, loading the model)
# to keep the app fast.
# ---------------------------------------------------------------
@st.cache_resource
def load_trained_model_artifacts():
    """Load the trained model, scaler, feature list, and metadata."""
    trained_model = joblib.load("models/best_model.joblib")
    feature_scaler = joblib.load("models/feature_scaler.joblib")
    feature_columns = joblib.load("models/feature_columns.joblib")
    model_metadata = joblib.load("models/model_metadata.joblib")
    return trained_model, feature_scaler, feature_columns, model_metadata


@st.cache_data
def load_real_patient_data():
    """Load the real (untouched) Pima patient dataset for EDA pages."""
    real_dataframe = pd.read_csv("data/pima_diabetes_real.csv")
    return real_dataframe


@st.cache_data
def load_model_comparison_table():
    """Load the saved comparison table produced by train_model.py."""
    comparison_dataframe = pd.read_csv("models/model_comparison_results.csv")
    return comparison_dataframe


def build_feature_row_from_user_input(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
    feature_columns,
):
    """
    Take raw values typed by the user, run them through the exact same
    feature-engineering steps used during training, and return a
    single-row dataframe with columns matching the trained model.
    """
    raw_input_dataframe = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree_function,
                "Age": age,
            }
        ]
    )

    engineered_dataframe = preprocessing.engineer_new_features(raw_input_dataframe)

    categorical_columns = ["BMI_Category", "Age_Group", "Glucose_Category"]
    encoded_dataframe = preprocessing.one_hot_encode_categorical_features(
        engineered_dataframe, categorical_columns
    )

    # Make sure every column the model expects is present, in the
    # right order, filling any missing one-hot column with 0.
    final_feature_row = encoded_dataframe.reindex(columns=feature_columns, fill_value=0)
    return final_feature_row


def get_risk_level_label(probability_of_diabetes):
    """Convert a raw probability into a friendly risk level label."""
    if probability_of_diabetes < 0.30:
        return "Low Risk", "green"
    elif probability_of_diabetes < 0.60:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"


# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
st.sidebar.title("🩺 Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ["Home", "Disease Prediction", "Dataset Insights", "Model Performance", "Graph Dashboard", "About"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This tool is an educational demo built with classic machine learning. "
    "It is NOT a certified medical device and must never replace a doctor."
)


# ---------------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------------
if selected_page == "Home":
    st.title("🩺 AI-Powered Diabetes Diagnosis System")
    st.subheader("An end-to-end Machine Learning project for early diabetes risk assessment")

    st.markdown(
        """
        Welcome! This application predicts a patient's **diabetes risk** using
        classic, interpretable machine learning models trained on real clinical
        data, following the full data science lifecycle:

        1. **Data collection** — real clinical data plus a statistically
           faithful synthetic expansion for training at scale.
        2. **Data cleaning & feature engineering** — handling missing values,
           removing outliers, and creating clinically meaningful features.
        3. **Model comparison** — 8 machine learning algorithms trained and
           evaluated head-to-head.
        4. **Deployment** — the best model is served here, through this
           interactive Streamlit app.
        """
    )

    trained_model, feature_scaler, feature_columns, model_metadata = load_trained_model_artifacts()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Best Model", model_metadata["best_model_name"])
    metric_col2.metric("Real-World Accuracy", f"{model_metadata['real_world_accuracy'] * 100:.1f}%")
    metric_col3.metric("Real-World ROC-AUC", f"{model_metadata['real_world_roc_auc']:.3f}")

    st.markdown("---")
    st.markdown(
        "Use the sidebar to navigate to **Disease Prediction** to try the model, "
        "or explore **Dataset Insights** and **Model Performance** to see the analysis behind it."
    )


# ---------------------------------------------------------------
# PAGE 2: DISEASE PREDICTION
# ---------------------------------------------------------------
elif selected_page == "Disease Prediction":
    st.title("🔍 Diabetes Risk Prediction")
    st.markdown("Fill in the patient's clinical measurements below, then click **Predict**.")

    trained_model, feature_scaler, feature_columns, model_metadata = load_trained_model_artifacts()

    with st.form("patient_input_form"):
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.slider("Glucose Level (mg/dL)", min_value=0, max_value=300, value=110)
            blood_pressure = st.slider("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70)
            skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

        with input_col2:
            insulin = st.slider("Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
            bmi = st.slider("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree_function = st.slider(
                "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.4, step=0.01
            )
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_feature_row = build_feature_row_from_user_input(
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age,
            feature_columns,
        )

        scaled_input_row = feature_scaler.transform(input_feature_row)

        predicted_class = trained_model.predict(scaled_input_row)[0]

        if hasattr(trained_model, "predict_proba"):
            probability_of_diabetes = trained_model.predict_proba(scaled_input_row)[0, 1]
        else:
            # Fallback for models without predict_proba (not used by
            # the default best model, but kept here for robustness).
            decision_score = trained_model.decision_function(scaled_input_row)[0]
            probability_of_diabetes = 1 / (1 + np.exp(-decision_score))

        risk_label, risk_color = get_risk_level_label(probability_of_diabetes)

        st.markdown("---")
        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            prediction_text = "Diabetic" if predicted_class == 1 else "Non-Diabetic"
            st.metric("Prediction", prediction_text)

        with result_col2:
            st.metric("Confidence Score", f"{max(probability_of_diabetes, 1 - probability_of_diabetes) * 100:.1f}%")

        with result_col3:
            st.markdown(f"**Risk Level:** :{risk_color}[{risk_label}]")

        gauge_figure = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_of_diabetes * 100,
                title={"text": "Diabetes Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 60], "color": "gold"},
                        {"range": [60, 100], "color": "salmon"},
                    ],
                },
            )
        )
        st.plotly_chart(gauge_figure, use_container_width=True)

        st.warning(
            "This prediction is generated by a statistical model for educational purposes only. "
            "Always consult a qualified healthcare professional for an actual diagnosis."
        )


# ---------------------------------------------------------------
# PAGE 3: DATASET INSIGHTS
# ---------------------------------------------------------------
elif selected_page == "Dataset Insights":
    st.title("📊 Dataset Insights")
    st.markdown(
        "These charts are based on the **real** 768-patient Pima Indians Diabetes "
        "Dataset (NIDDK), kept completely separate from the synthetic training data."
    )

    real_dataframe = load_real_patient_data()

    st.subheader("Preview of the Real Dataset")
    st.dataframe(real_dataframe.head(10))

    st.subheader("Summary Statistics")
    st.dataframe(real_dataframe.describe())

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.subheader("Outcome Class Balance")
        outcome_counts = real_dataframe["Outcome"].value_counts().rename({0: "Non-Diabetic", 1: "Diabetic"})
        count_plot_figure, count_plot_axes = plt.subplots()
        sns.countplot(x=real_dataframe["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}), ax=count_plot_axes)
        count_plot_axes.set_xlabel("Outcome")
        count_plot_axes.set_ylabel("Number of Patients")
        st.pyplot(count_plot_figure)

    with insight_col2:
        st.subheader("Glucose Distribution")
        histogram_figure, histogram_axes = plt.subplots()
        sns.histplot(real_dataframe["Glucose"], kde=True, ax=histogram_axes)
        histogram_axes.set_xlabel("Glucose Level")
        st.pyplot(histogram_figure)

    st.subheader("Correlation Heatmap")
    heatmap_figure, heatmap_axes = plt.subplots(figsize=(9, 6))
    sns.heatmap(real_dataframe.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=heatmap_axes)
    st.pyplot(heatmap_figure)

    st.subheader("BMI vs Glucose, colored by Outcome (interactive)")
    scatter_figure = px.scatter(
        real_dataframe,
        x="BMI",
        y="Glucose",
        color=real_dataframe["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
        labels={"color": "Outcome"},
        title="BMI vs Glucose",
    )
    st.plotly_chart(scatter_figure, use_container_width=True)


# ---------------------------------------------------------------
# PAGE 4: MODEL PERFORMANCE
# ---------------------------------------------------------------
elif selected_page == "Model Performance":
    st.title("📈 Model Performance Comparison")

    comparison_dataframe = load_model_comparison_table()
    trained_model, feature_scaler, feature_columns, model_metadata = load_trained_model_artifacts()

    st.markdown(f"**Selected best model:** `{model_metadata['best_model_name']}`")
    st.dataframe(comparison_dataframe.round(4))

    st.subheader("Real-World F1 Score by Model")
    bar_figure = px.bar(
        comparison_dataframe.sort_values("real_f1", ascending=False),
        x="model_name",
        y="real_f1",
        color="real_f1",
        color_continuous_scale="Blues",
        labels={"model_name": "Model", "real_f1": "Real-World F1 Score"},
    )
    st.plotly_chart(bar_figure, use_container_width=True)

    st.subheader("Real-World Accuracy vs ROC-AUC")
    scatter_figure = px.scatter(
        comparison_dataframe,
        x="real_accuracy",
        y="real_roc_auc",
        text="model_name",
        labels={"real_accuracy": "Real-World Accuracy", "real_roc_auc": "Real-World ROC-AUC"},
    )
    scatter_figure.update_traces(textposition="top center")
    st.plotly_chart(scatter_figure, use_container_width=True)


# ---------------------------------------------------------------
# PAGE 5: GRAPH DASHBOARD
# ---------------------------------------------------------------
elif selected_page == "Graph Dashboard":
    st.title("📉 Graph Dashboard")
    st.markdown("Extra interactive visualizations for deeper exploration of the real dataset.")

    real_dataframe = load_real_patient_data()

    selected_feature = st.selectbox(
        "Choose a feature to explore",
        [column for column in real_dataframe.columns if column != "Outcome"],
    )

    box_plot_figure = px.box(
        real_dataframe,
        x=real_dataframe["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
        y=selected_feature,
        color=real_dataframe["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"}),
        labels={"x": "Outcome"},
        title=f"{selected_feature} by Outcome",
    )
    st.plotly_chart(box_plot_figure, use_container_width=True)

    st.subheader("Pairwise Feature Relationships")
    pairplot_columns = st.multiselect(
        "Choose up to 4 features for the pair plot",
        [column for column in real_dataframe.columns if column != "Outcome"],
        default=["Glucose", "BMI", "Age"],
    )

    if len(pairplot_columns) >= 2:
        pairplot_dataframe = real_dataframe[pairplot_columns + ["Outcome"]].copy()
        pairplot_dataframe["Outcome"] = pairplot_dataframe["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})
        pair_plot_figure = sns.pairplot(pairplot_dataframe, hue="Outcome", diag_kind="kde")
        st.pyplot(pair_plot_figure)
    else:
        st.info("Select at least 2 features to draw a pair plot.")


# ---------------------------------------------------------------
# PAGE 6: ABOUT
# ---------------------------------------------------------------
elif selected_page == "About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        **Project:** AI-Powered Medical Diagnosis System (Diabetes)

        **Technology stack:** Python, NumPy, Pandas, Matplotlib, Seaborn,
        Plotly, Scikit-learn, XGBoost, and Streamlit — no deep learning
        frameworks were used.

        **Dataset:** Pima Indians Diabetes Dataset (NIDDK), 768 real
        patient records, expanded with a statistically-faithful
        synthetic dataset for training at scale. See `README.md` for
        full details and honest disclosure of how the synthetic data
        was generated.

        **Disclaimer:** This application is an educational portfolio
        project. It is not a certified medical device and should never
        be used as a substitute for professional medical advice,
        diagnosis, or treatment.
        """
    )
