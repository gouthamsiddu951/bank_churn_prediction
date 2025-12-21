import os
import joblib

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_artifacts(model, model_path: str, preproc_path: str, preprocessor):
    ensure_dir(model_path)
    ensure_dir(preproc_path)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)
    print(f"Saved model to {model_path} and preprocessor to {preproc_path}")

def load_artifacts(model_path: str, preproc_path: str):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    return model, preprocessor
