https://bankchurnprediction-gafo6gtniehswtvzlsifwh.streamlit.app/

Thia is the public link that can be accesed.
📌 Project Overview
Title: Bank Customer Churn Prediction

Goal: Predict whether a customer will leave the bank (churn) based on demographic, financial, and behavioral features.

Tech Stack: Python, Streamlit, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn.

Deployment: Interactive web app built with Streamlit, deployable on Streamlit Cloud.
⚙️ Features
Interactive Prediction: Users can input customer details and get churn probability.

Explainability (SHAP): Visual explanations of feature contributions for individual predictions.

Dashboard Analytics: Dataset-level insights (churn by geography, age, tenure, etc.).

Threshold Control: Adjustable decision threshold to balance precision vs recall.

Visualization: Clear plots for churn distribution, feature importance, and decision paths.
📊 Dataset
Source: Churn_Modelling.csv (10,000 customers).

Features: Credit score, geography, gender, age, tenure, balance, products, credit card ownership, activity status, salary.

Target: Exited (1 = churn, 0 = stay).
🛠️ How It Works
Preprocessing: Encodes categorical variables, scales numerical features.

Models:

XGBoost pipeline → used for SHAP explanations.

Ensemble pipeline → used for predictions (higher accuracy).

Streamlit App:

Sidebar navigation: Predict | Explain (SHAP) | Dashboard.

User-friendly input forms.

Real-time visualizations.
