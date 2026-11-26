
# Loan Status Prediction Project - README

## Project Overview
This project aims to predict loan approval/default status (`loan_status`) using customer demographic, financial, and credit history features. Multiple supervised machine learning algorithms were trained, evaluated, and compared to select the best-performing model.

## Dataset Description
The dataset contains **45,000 rows** and **14 columns** initially, featuring:
- **Numerical Features**: `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`.
- **Categorical Features**: `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`.
- **Target**: `loan_status` (Binary: 0 for non-default, 1 for default).

## Data Preprocessing Workflow
1. **Data Cleaning**: Confirmed no missing values and 0 duplicates.
2. **Feature Dropping**: Irrelevant or highly subjective features (`person_gender`, `person_education`, `person_age`) were removed to avoid bias.
3. **Outlier Filtering**: Customer employment experience (`person_emp_exp`) values above 40 years were filtered out as unrealistic anomalies.
4. **Categorical Encoding**: Used `LabelEncoder` to encode non-numeric categorical variables.
5. **Stratified Split**: Split the dataset into 80% Training and 20% Testing sets using `stratify=y` to preserve the imbalanced class ratio (~77.8% non-default, ~22.2% default).
6. **Feature Scaling**: Applied `StandardScaler` on numerical columns (`person_income`, `loan_amnt`, `credit_score`) to prevent scale-related issues during model training.

## Models Trained
- Logistic Regression
- K-Nearest Neighbors (KNN) (Manhattan metric, k=4)
- Decision Tree (ID3 - Entropy, Max Depth = 6)
- Decision Tree (CART - Gini, Max Depth = 6)
- Naive Bayes (Gaussian)
- Multi-Layer Perceptron (MLP) Neural Network (Hidden Layers: 64x32)

## Performance Comparison & Evaluation
All models were evaluated across standard classification metrics on the test dataset:

| Model | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 84.10% | 0.689 | 0.520 | 0.593 | 0.858 |
| **KNN** | 86.08% | 0.785 | 0.515 | 0.622 | 0.878 |
| **Naive Bayes** | 81.60% | 0.727 | 0.276 | 0.400 | 0.809 |
| **ID3 Decision Tree** | 91.30% | 0.886 | 0.698 | 0.781 | 0.959 |
| **CART Decision Tree** | 91.41% | 0.881 | 0.709 | 0.786 | 0.958 |
| **Multi-Layer Perceptron (MLP)** | **91.51%** | **0.853** | **0.747** | **0.796** | **0.966** |

## Final Model Selection
The **Multi-Layer Perceptron (MLP)** was selected as the champion model. It displays strong robustness across all metrics, achieving the highest **Accuracy (91.51%)**, **Recall (74.67%)**, **F1-Score (0.796)**, and **ROC-AUC score (0.966)**, making it highly effective at identifying defaults while maintaining a high precision rate.

