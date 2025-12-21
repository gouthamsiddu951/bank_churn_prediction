import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

FEATURES_TO_DROP = ["RowNumber", "CustomerId", "Surname"]
TARGET_COL = "Exited"

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_X_y(df: pd.DataFrame):
    df = df.drop(columns=FEATURES_TO_DROP)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
