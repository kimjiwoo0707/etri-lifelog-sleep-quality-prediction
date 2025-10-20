import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

common_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 10,
    "num_leaves": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def cross_val_predict_average(X, y, test_X, model_fn, n_splits=5):

    preds = np.zeros((len(test_X),))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f" Fold {fold+1}/{n_splits} 시작")
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds += model.predict_proba(test_X)[:, 1]
        print(f" Fold {fold+1} 완료")

    return preds / n_splits


def get_voting_model_logistic():

    lgbm = LGBMClassifier(**common_params)

    xgb_params = common_params.copy()
    xgb_params.pop("verbosity", None)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **xgb_params)

    cat_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 10,
        "subsample": 0.8,
        "random_state": 42,
        "verbose": 0,
    }
    cat = CatBoostClassifier(**cat_params)

    logreg = LogisticRegression(max_iter=1000)

    return VotingClassifier(
        estimators=[
            ("lgbm", lgbm),
            ("xgb", xgb),
            ("cat", cat),
            ("logreg", logreg),
        ],
        voting="soft",
        n_jobs=-1,
    )


def get_stacking_preds(X, y, model_fn, n_splits=5):

    meta_preds = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"🔁 Stacking Fold {fold+1}/{n_splits} 시작")
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        meta_preds[val_idx] = model.predict(X.iloc[val_idx])
        print(f"✅ Fold {fold+1} 완료")

    return meta_preds
