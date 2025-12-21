import pandas as pd
from src.utils import load_artifacts

# Choose which pipeline to load: ensemble is default for production
PIPELINE_PATH = "artifacts/ensemble_pipeline.joblib"
PREPROC_PATH = "artifacts/preprocessor.joblib"

def predict_single(sample_dict, threshold=0.5):
    model, _ = load_artifacts(PIPELINE_PATH, PREPROC_PATH)
    df = pd.DataFrame([sample_dict])
    proba = model.predict_proba(df)[0, 1]
    label = int(proba >= threshold)
    return proba, label

if __name__ == "__main__":
    sample = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 5,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    proba, label = predict_single(sample, threshold=0.45)
    print(f"Churn probability: {proba:.4f} | Predicted: {'Churn' if label==1 else 'Stay'}")
