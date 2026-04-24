import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.utils import load_artifacts
from src.preprocess import load_data, split_X_y

# Paths
XGB_PIPELINE_PATH = "artifacts/xgb_pipeline.joblib"
ENSEMBLE_PIPELINE_PATH = "artifacts/ensemble_pipeline.joblib"
PREPROC_PATH = "artifacts/preprocessor.joblib"
DATA_PATH = "data/Churn_Modelling.csv"

# Page config
st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

# Header
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("""
### 🔍 Predict customer churn risk using Machine Learning  
- Real-time predictions  
- Explainable AI (SHAP)  
- Business insights dashboard  
""")
st.caption("Interactive predictions, explanations (SHAP), and a dashboard with insights.")

# Load models
xgb_pipeline, _ = load_artifacts(XGB_PIPELINE_PATH, PREPROC_PATH)
ensemble_pipeline, _ = load_artifacts(ENSEMBLE_PIPELINE_PATH, PREPROC_PATH)

# Sidebar
view = st.sidebar.radio("View", ["Predict", "Explain (SHAP)", "Dashboard"])

# Input form
def input_form():
    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.number_input("Customer ID", min_value=1, value=123456)
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure", 0, 10, 5)

    with col2:
        balance = st.number_input("Balance", 0.0, value=60000.0)
        num_products = st.number_input("Products", 1, 4, 2)
        has_card = st.selectbox("Has Credit Card", [0, 1])
        is_active = st.selectbox("Is Active Member", [0, 1])
        salary = st.number_input("Estimated Salary", 0.0, value=50000.0)

    sample = {
        "CustomerId": int(customer_id),
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


# ============================
# 🔮 PREDICT
# ============================
if view == "Predict":
    st.subheader("Make a prediction")
    df = input_form()

    if st.button("Predict"):

        customer_id = df["CustomerId"].values[0]
        df_model = df.drop(columns=["CustomerId"])

        with st.spinner("Predicting..."):
            proba = ensemble_pipeline.predict_proba(df_model)[0, 1]

        st.metric("Churn Probability", f"{proba:.2f}")
        st.write(f"Customer ID: {customer_id}")

        # Label
        label = "Churn" if proba >= 0.5 else "Stay"

        # Risk Levels
        if proba < 0.4:
            st.success(f"🟢 Low Risk ({label})")
        elif proba < 0.7:
            st.warning(f"🟡 Medium Risk ({label})")
        else:
            st.error(f"🔴 High Risk ({label})")

        # Business Recommendation
        st.subheader("💡 Recommendation")
        if proba > 0.7:
            st.write("Offer discounts or retention plans immediately.")
        elif proba > 0.4:
            st.write("Engage with personalized offers.")
        else:
            st.write("Customer is stable.")

        # Why prediction
        st.subheader("🔍 Why this prediction?")
        st.write("""
        - Higher age increases churn risk  
        - Inactive members churn more  
        - Low product usage impacts retention  
        """)

        # Batch upload
        st.divider()
        st.subheader("📂 Batch Prediction")

        uploaded = st.file_uploader("Upload CSV")

        if uploaded:
            data = pd.read_csv(uploaded)
            preds = ensemble_pipeline.predict_proba(data)[:, 1]
            data["Churn_Probability"] = preds
            st.write(data)

            st.download_button(
                "Download Results",
                data=data.to_csv(index=False),
                file_name="predictions.csv"
            )


# ============================
# 🔍 SHAP EXPLAIN
# ============================
elif view == "Explain (SHAP)":
    st.subheader("Explain Prediction")
    df = input_form()

    if st.button("Explain"):

        xgb_model = xgb_pipeline.named_steps["model"]
        X_transformed = xgb_pipeline.named_steps["preproc"].transform(df)

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        st.write("Feature Contributions:")

        preproc = xgb_pipeline.named_steps["preproc"]
        num_cols = preproc.transformers_[0][2]
        cat_cols = preproc.transformers_[1][2]
        ohe = preproc.transformers_[1][1].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([num_cols, ohe_names])

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig)


# ============================
# 📊 DASHBOARD
# ============================
else:
    st.subheader("📊 Dashboard")

    df_raw = load_data(DATA_PATH)
    X, y = split_X_y(df_raw)

    df_dash = X.copy()
    df_dash["Exited"] = y

    # KPI Cards
    churn_rate = df_dash["Exited"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(df_dash))
    col2.metric("Churn Rate", f"{churn_rate:.2%}")
    col3.metric("Avg Age", int(df_dash["Age"].mean()))

    # Filter
    geo = st.selectbox("Filter by Geography", df_dash["Geography"].unique())
    df_dash = df_dash[df_dash["Geography"] == geo]

    # Charts
    st.write("Churn by Geography")
    geo_churn = df_dash.groupby("Geography")["Exited"].mean()

    fig, ax = plt.subplots()
    geo_churn.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.write("Churn by Tenure")
    ten_churn = df_dash.groupby("Tenure")["Exited"].mean()

    fig2, ax2 = plt.subplots()
    ten_churn.plot(kind="line", marker="o", ax=ax2)
    st.pyplot(fig2)

    # Insight
    st.subheader("📌 Insight")
    st.write("Inactive customers and older users show higher churn rates.")


# Footer
st.markdown("---")
st.caption("Developed by M Gowtham Kumar 🚀")