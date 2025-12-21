from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

def logistic_baseline():
    return LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)

def rf_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample"
    )

def xgb_strong():
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

def soft_voting_ensemble():
    return VotingClassifier(
        estimators=[
            ("log", logistic_baseline()),
            ("rf", rf_model()),
            ("xgb", xgb_strong()),
        ],
        voting="soft"
    )
