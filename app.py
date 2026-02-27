import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.utils import load_artifacts
from src.preprocess import load_data, split_X_y

# Paths
XGB_PIPELINE_PATH = "artifacts/xgb_pipeline.joblib"       # For SHAP
ENSEMBLE_PIPELINE_PATH = "artifacts/ensemble_pipeline.joblib"  # For predictions
PREPROC_PATH = "artifacts/preprocessor.joblib"
DATA_PATH = "data/Churn_Modelling.csv"

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

st.title("🏦 Bank Customer Churn Prediction")
st.caption("Interactive predictions, explanations (SHAP), and a dashboard with segment insights.")

# Load pipelines
xgb_pipeline, _ = load_artifacts(XGB_PIPELINE_PATH, PREPROC_PATH)
ensemble_pipeline, _ = load_artifacts(ENSEMBLE_PIPELINE_PATH, PREPROC_PATH)

# Sidebar navigation
view = st.sidebar.radio("View", ["Predict", "Explain (SHAP)", "Dashboard"])

# Common input widget factory
def input_form():
    col1, col2 = st.columns(2)
    with col1:
        customer_id = st.number_input("Customer ID", min_value=1, value=123456)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10, value=5)
    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=100.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
        has_card = st.selectbox("Has Credit Card", [0, 1], index=1)
        is_active = st.selectbox("Is Active Member", [0, 1], index=1)
        salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0)

    sample = {
    "CustomerId": int(customer_id),   # NEW
    "CreditScore": int(credit_score),
    "Geography": geography,
    "Gender": gender,
    "Age": int(age),
    "Tenure": int(tenure),
    "Balance": float(balance),
    "NumOfProducts": int(num_products),
    "HasCrCard": int(has_card),
    "IsActiveMember": int(is_active),
    "EstimatedSalary": float(salary)
}

    return pd.DataFrame([sample])


# Predict view (uses ensemble for better accuracy)
if view == "Predict":
    st.subheader("Make a prediction")
    df = input_form()
    if st.button("Predict with Ensemble"):
        customer_id = df["CustomerId"].values[0]

        # Remove CustomerId before prediction
        df_model = df.drop(columns=["CustomerId"])

        proba = ensemble_pipeline.predict_proba(df_model)[0, 1]
        label = "Churn" if proba >= 0.5 else "Stay"

        st.metric("Churn probability", f"{proba:.2f}")
        st.write(f"Customer ID: {customer_id}")

        if label == "Churn":
            st.error(f"🚨 Prediction: {label}")
        else:
            st.success(f"✅ Prediction: {label}")
        label = "Churn" if proba >= 0.5 else "Stay"        # fixed cutoff at 0.5
        st.metric("Churn probability", f"{proba:.2f}")
        if label == "Churn":
            st.error(f"🚨 Prediction: {label}")
        else:
            st.success(f"✅ Prediction: {label}")


# Explain view (SHAP; uses XGB for compatibility)
elif view == "Explain (SHAP)":
    st.subheader("Explain a single prediction (SHAP)")
    st.caption("Using the XGBoost pipeline for SHAP explanations.")
    df = input_form()

    if st.button("Explain with SHAP"):
        # Extract the trained XGB model from the pipeline
        xgb_model = xgb_pipeline.named_steps["model"]
        # Preprocess the single row to match training features
        X_transformed = xgb_pipeline.named_steps["preproc"].transform(df)

        # Build SHAP explainer (TreeExplainer works for XGB)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        st.write("Feature contributions to prediction:")

        # Build a readable feature names list from the ColumnTransformer
        preproc = xgb_pipeline.named_steps["preproc"]
        num_cols = preproc.transformers_[0][2]
        cat_cols = preproc.transformers_[1][2]
        ohe = preproc.transformers_[1][1].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([num_cols, ohe_names])

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig)

        # Decision plot for the single prediction
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        shap.decision_plot(explainer.expected_value, shap_values[0], feature_names=feature_names, show=False)
        st.pyplot(fig2)

# Dashboard view (dataset-level insights)
else:
    st.subheader("Dataset dashboard")
    df_raw = load_data(DATA_PATH)
    X, y = split_X_y(df_raw)
    df_dash = X.copy()
    df_dash["Exited"] = y

    # High-level churn rate
    churn_rate = df_dash["Exited"].mean()
    st.metric("Overall churn rate", f"{churn_rate:.2%}")

    # Churn by geography
    st.write("Churn by geography")
    geo_churn = df_dash.groupby("Geography")["Exited"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=geo_churn.index, y=geo_churn.values, ax=ax)
    ax.set_ylabel("Churn rate")
    st.pyplot(fig)

    # Churn by age buckets
    st.write("Churn by age bucket")
    age_bins = [18, 30, 40, 50, 60, 100]
    df_dash["age_bucket"] = pd.cut(df_dash["Age"], bins=age_bins)
    age_churn = df_dash.groupby("age_bucket")["Exited"].mean()
    fig2, ax2 = plt.subplots(figsize=(6,3))
    age_churn.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Churn rate")
    st.pyplot(fig2)

    # Churn vs tenure
    st.write("Churn by tenure")
    ten_churn = df_dash.groupby("Tenure")["Exited"].mean()
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ten_churn.plot(kind="line", marker="o", ax=ax3)
    ax3.set_ylabel("Churn rate")
    ax3.set_xlabel("Tenure (years)")
    st.pyplot(fig3)

    # Optional: probability distribution with ensemble
    st.write("Predicted probability distribution (ensemble)")
    probs = ensemble_pipeline.predict_proba(X)[:, 1]
    fig4, ax4 = plt.subplots(figsize=(6,3))
    sns.histplot(probs, bins=30, kde=True, ax=ax4)
    ax4.set_xlabel("Predicted churn probability")
    st.pyplot(fig4)
