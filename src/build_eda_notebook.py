"""
build_eda_notebook.py
--------------------------------------------------------------------
Builds notebooks/EDA_Diabetes_Analysis.ipynb programmatically, one
cell per analysis question, then executes it so every output is
already populated when the user opens it.
"""

import nbformat as nbf

notebook = nbf.v4.new_notebook()
cells = []

# ---------------------------------------------------------------
# Intro markdown cell
# ---------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        "# Exploratory Data Analysis — Diabetes Diagnosis Dataset\n"
        "This notebook walks through data loading, cleaning, feature engineering, "
        "and visualization for the diabetes prediction project. Each code cell "
        "answers one specific analysis question, stated in a comment at the top "
        "of the cell."
    )
)

code_cells_source = []

code_cells_source.append(
    "# Question 1: Import the libraries we need for this analysis\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import plotly.express as px\n"
    "\n"
    "sns.set_style('whitegrid')\n"
    "print('Libraries imported successfully')"
)

code_cells_source.append(
    "# Question 2: Load the large combined dataset (real + synthetic) and show the first 5 rows\n"
    "diabetes_dataframe = pd.read_csv('../data/diabetes_large_dataset.csv')\n"
    "diabetes_dataframe.head()"
)

code_cells_source.append(
    "# Question 3: What is the shape of the dataset (rows and columns)?\n"
    "number_of_rows = diabetes_dataframe.shape[0]\n"
    "number_of_columns = diabetes_dataframe.shape[1]\n"
    "print('Number of rows:', number_of_rows)\n"
    "print('Number of columns:', number_of_columns)"
)

code_cells_source.append(
    "# Question 4: What are the column names and data types?\n"
    "diabetes_dataframe.info()"
)

code_cells_source.append(
    "# Question 5: How many rows come from the real dataset versus the synthetic dataset?\n"
    "data_source_counts = diabetes_dataframe['data_source'].value_counts()\n"
    "print(data_source_counts)"
)

code_cells_source.append(
    "# Question 6: Are there any missing (NaN) values in the raw dataset?\n"
    "missing_value_counts = diabetes_dataframe.isnull().sum()\n"
    "print(missing_value_counts)"
)

code_cells_source.append(
    "# Question 7: Some columns use 0 to mean 'missing' instead of a real medical value.\n"
    "# How many zero values does each of those columns actually contain?\n"
    "columns_where_zero_means_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n"
    "for column_name in columns_where_zero_means_missing:\n"
    "    zero_count = (diabetes_dataframe[column_name] == 0).sum()\n"
    "    print(column_name, 'has', zero_count, 'zero values')"
)

code_cells_source.append(
    "# Question 8: Replace those invalid zero values with NaN so we can handle them properly\n"
    "cleaned_dataframe = diabetes_dataframe.copy()\n"
    "for column_name in columns_where_zero_means_missing:\n"
    "    zero_mask = cleaned_dataframe[column_name] == 0\n"
    "    cleaned_dataframe.loc[zero_mask, column_name] = np.nan\n"
    "\n"
    "missing_value_counts_after_replacement = cleaned_dataframe.isnull().sum()\n"
    "print(missing_value_counts_after_replacement)"
)

code_cells_source.append(
    "# Question 9: Fill the missing values using the median of each column, grouped by Outcome\n"
    "for column_name in columns_where_zero_means_missing:\n"
    "    median_by_outcome = cleaned_dataframe.groupby('Outcome')[column_name].transform('median')\n"
    "    missing_mask = cleaned_dataframe[column_name].isna()\n"
    "    cleaned_dataframe.loc[missing_mask, column_name] = median_by_outcome[missing_mask]\n"
    "\n"
    "remaining_missing_values = cleaned_dataframe.isnull().sum().sum()\n"
    "print('Remaining missing values after filling:', remaining_missing_values)"
)

code_cells_source.append(
    "# Question 10: Are there any duplicate rows in the dataset?\n"
    "number_of_duplicate_rows = cleaned_dataframe.duplicated().sum()\n"
    "print('Number of duplicate rows:', number_of_duplicate_rows)\n"
    "cleaned_dataframe = cleaned_dataframe.drop_duplicates()\n"
    "print('Shape after removing duplicates:', cleaned_dataframe.shape)"
)

code_cells_source.append(
    "# Question 11: How many outliers does the Glucose column contain, using the IQR method?\n"
    "first_quartile = cleaned_dataframe['Glucose'].quantile(0.25)\n"
    "third_quartile = cleaned_dataframe['Glucose'].quantile(0.75)\n"
    "interquartile_range = third_quartile - first_quartile\n"
    "lower_bound = first_quartile - 1.5 * interquartile_range\n"
    "upper_bound = third_quartile + 1.5 * interquartile_range\n"
    "outlier_mask = (cleaned_dataframe['Glucose'] < lower_bound) | (cleaned_dataframe['Glucose'] > upper_bound)\n"
    "number_of_outliers = outlier_mask.sum()\n"
    "print('Lower bound:', lower_bound)\n"
    "print('Upper bound:', upper_bound)\n"
    "print('Number of Glucose outliers:', number_of_outliers)"
)

code_cells_source.append(
    "# Question 12: Remove outliers from every key numeric column using the IQR method\n"
    "columns_to_check_for_outliers = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']\n"
    "rows_before_outlier_removal = cleaned_dataframe.shape[0]\n"
    "\n"
    "for column_name in columns_to_check_for_outliers:\n"
    "    first_quartile = cleaned_dataframe[column_name].quantile(0.25)\n"
    "    third_quartile = cleaned_dataframe[column_name].quantile(0.75)\n"
    "    interquartile_range = third_quartile - first_quartile\n"
    "    lower_bound = first_quartile - 1.5 * interquartile_range\n"
    "    upper_bound = third_quartile + 1.5 * interquartile_range\n"
    "    within_bounds_mask = cleaned_dataframe[column_name].between(lower_bound, upper_bound)\n"
    "    cleaned_dataframe = cleaned_dataframe[within_bounds_mask]\n"
    "\n"
    "cleaned_dataframe = cleaned_dataframe.reset_index(drop=True)\n"
    "rows_after_outlier_removal = cleaned_dataframe.shape[0]\n"
    "print('Rows before outlier removal:', rows_before_outlier_removal)\n"
    "print('Rows after outlier removal:', rows_after_outlier_removal)"
)

code_cells_source.append(
    "# Question 13: Create a BMI Category feature using standard WHO BMI bands\n"
    "bmi_bin_edges = [0, 18.5, 24.9, 29.9, 100]\n"
    "bmi_bin_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']\n"
    "cleaned_dataframe['BMI_Category'] = pd.cut(cleaned_dataframe['BMI'], bins=bmi_bin_edges, labels=bmi_bin_labels)\n"
    "bmi_category_counts = cleaned_dataframe['BMI_Category'].value_counts()\n"
    "print(bmi_category_counts)"
)

code_cells_source.append(
    "# Question 14: Create an Age Group feature\n"
    "age_bin_edges = [0, 30, 45, 60, 120]\n"
    "age_bin_labels = ['Young', 'Middle_Aged', 'Senior', 'Elderly']\n"
    "cleaned_dataframe['Age_Group'] = pd.cut(cleaned_dataframe['Age'], bins=age_bin_edges, labels=age_bin_labels)\n"
    "age_group_counts = cleaned_dataframe['Age_Group'].value_counts()\n"
    "print(age_group_counts)"
)

code_cells_source.append(
    "# Question 15: What is the class balance of the target variable (Outcome)?\n"
    "outcome_counts = cleaned_dataframe['Outcome'].value_counts()\n"
    "outcome_percentages = cleaned_dataframe['Outcome'].value_counts(normalize=True) * 100\n"
    "print('Outcome counts:')\n"
    "print(outcome_counts)\n"
    "print('Outcome percentages:')\n"
    "print(outcome_percentages)"
)

code_cells_source.append(
    "# Question 16: What does the correlation matrix look like for the numeric features?\n"
    "numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']\n"
    "correlation_matrix = cleaned_dataframe[numeric_columns].corr()\n"
    "correlation_matrix"
)

code_cells_source.append(
    "# Question 17: Which features correlate most strongly with the Outcome?\n"
    "correlation_with_outcome = correlation_matrix['Outcome'].sort_values(ascending=False)\n"
    "print(correlation_with_outcome)"
)

code_cells_source.append(
    "# Question 18: What does the distribution of Glucose look like? (Matplotlib histogram)\n"
    "plt.figure(figsize=(8, 5))\n"
    "plt.hist(cleaned_dataframe['Glucose'], bins=30, color='steelblue', edgecolor='black')\n"
    "plt.title('Distribution of Glucose Levels')\n"
    "plt.xlabel('Glucose')\n"
    "plt.ylabel('Number of Patients')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 19: What does the distribution of BMI look like? (Seaborn histogram with KDE)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.histplot(cleaned_dataframe['BMI'], kde=True, color='seagreen')\n"
    "plt.title('Distribution of BMI')\n"
    "plt.xlabel('BMI')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 20: What does the distribution of Age look like?\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.histplot(cleaned_dataframe['Age'], kde=True, color='indianred')\n"
    "plt.title('Distribution of Age')\n"
    "plt.xlabel('Age')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 21: How does Glucose differ between diabetic and non-diabetic patients? (Box plot)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.boxplot(x='Outcome', y='Glucose', data=cleaned_dataframe)\n"
    "plt.title('Glucose Level by Outcome')\n"
    "plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 22: How does BMI differ between diabetic and non-diabetic patients? (Box plot)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.boxplot(x='Outcome', y='BMI', data=cleaned_dataframe)\n"
    "plt.title('BMI by Outcome')\n"
    "plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 23: What does the full correlation heatmap look like?\n"
    "plt.figure(figsize=(10, 7))\n"
    "sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')\n"
    "plt.title('Correlation Heatmap')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 24: Is there a relationship between Glucose and BMI? (Scatter plot)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=cleaned_dataframe, alpha=0.5)\n"
    "plt.title('Glucose vs BMI, colored by Outcome')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 25: Is there a relationship between Age and Glucose? (Scatter plot)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=cleaned_dataframe, alpha=0.5)\n"
    "plt.title('Glucose vs Age, colored by Outcome')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 26: How many patients fall into each BMI Category? (Count plot)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.countplot(x='BMI_Category', data=cleaned_dataframe, order=['Underweight', 'Normal', 'Overweight', 'Obese'])\n"
    "plt.title('Number of Patients per BMI Category')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 27: How many patients fall into each Age Group? (Count plot)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.countplot(x='Age_Group', data=cleaned_dataframe, order=['Young', 'Middle_Aged', 'Senior', 'Elderly'])\n"
    "plt.title('Number of Patients per Age Group')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 28: What do the pairwise relationships between key features look like? (Pair plot)\n"
    "pairplot_dataframe = cleaned_dataframe[['Glucose', 'BMI', 'Age', 'Outcome']].copy()\n"
    "pairplot_sample = pairplot_dataframe.sample(n=2000, random_state=42)\n"
    "sns.pairplot(pairplot_sample, hue='Outcome', diag_kind='kde')\n"
    "plt.show()"
)

code_cells_source.append(
    "# Question 29: Build an interactive histogram of Glucose using Plotly\n"
    "plotly_histogram_figure = px.histogram(cleaned_dataframe, x='Glucose', color='Outcome', nbins=40, title='Interactive Glucose Distribution')\n"
    "plotly_histogram_figure.show()"
)

code_cells_source.append(
    "# Question 30: Build an interactive scatter plot of Glucose vs BMI using Plotly\n"
    "plotly_scatter_sample = cleaned_dataframe.sample(n=3000, random_state=42)\n"
    "plotly_scatter_figure = px.scatter(plotly_scatter_sample, x='BMI', y='Glucose', color='Outcome', title='Interactive Glucose vs BMI')\n"
    "plotly_scatter_figure.show()"
)

code_cells_source.append(
    "# Question 31: Build an interactive box plot comparing Insulin across Outcome groups using Plotly\n"
    "plotly_box_figure = px.box(cleaned_dataframe, x='Outcome', y='Insulin', color='Outcome', title='Interactive Insulin by Outcome')\n"
    "plotly_box_figure.show()"
)

code_cells_source.append(
    "# Question 32: Save the fully cleaned dataframe so it can be reused by the training script\n"
    "cleaned_dataframe.to_csv('../data/diabetes_cleaned_from_notebook.csv', index=False)\n"
    "print('Cleaned dataset saved with shape:', cleaned_dataframe.shape)"
)

for source in code_cells_source:
    cells.append(nbf.v4.new_code_cell(source))

notebook["cells"] = cells

with open("notebooks/EDA_Diabetes_Analysis.ipynb", "w") as notebook_file:
    nbf.write(notebook, notebook_file)

print("Notebook built with", len(cells), "cells")
