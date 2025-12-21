from src.preprocess import load_data, split_X_y, build_preprocessor
from src.models import xgb_strong, soft_voting_ensemble
from src.evaluate import evaluate
from src.utils import save_artifacts
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = "data/Churn_Modelling.csv"
# Save both models: XGB for SHAP explanations; Ensemble for production
XGB_MODEL_PATH = "artifacts/xgb_pipeline.joblib"
ENSEMBLE_MODEL_PATH = "artifacts/ensemble_pipeline.joblib"
PREPROC_PATH = "artifacts/preprocessor.joblib"

def main():
    df = load_data(DATA_PATH)
    X, y = split_X_y(df)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train strong XGB model
    xgb_model = xgb_strong()
    xgb_pipeline = Pipeline([("preproc", preprocessor), ("model", xgb_model)])
    xgb_pipeline.fit(X_train, y_train)
    evaluate(xgb_pipeline, X_test, y_test, title="XGBoost")

    # Train soft-voting ensemble
    ensemble_model = soft_voting_ensemble()
    ensemble_pipeline = Pipeline([("preproc", preprocessor), ("model", ensemble_model)])
    ensemble_pipeline.fit(X_train, y_train)
    evaluate(ensemble_pipeline, X_test, y_test, title="Soft Voting Ensemble")

    # Save artifacts
    save_artifacts(xgb_pipeline, XGB_MODEL_PATH, PREPROC_PATH, preprocessor)
    save_artifacts(ensemble_pipeline, ENSEMBLE_MODEL_PATH, PREPROC_PATH, preprocessor)

if __name__ == "__main__":
    main()
