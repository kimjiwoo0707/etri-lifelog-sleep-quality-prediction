import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

from .model import (
    common_params,
    cross_val_predict_average,
    get_voting_model_logistic,
)


def train_all_targets(X: pd.DataFrame, train_df: pd.DataFrame, test_X: pd.DataFrame):

    targets_binary = ["Q1", "Q2", "Q3", "S2", "S3"]
    target_multiclass = "S1"

    model_q2 = LGBMClassifier(**common_params)
    model_q2.fit(X, train_df["Q2"])

    feature_importance_q2 = (
        pd.DataFrame({"feature": X.columns, "importance": model_q2.feature_importances_})
        .sort_values(by="importance", ascending=False)
    )
    top_features_q2 = feature_importance_q2.head(30)["feature"].tolist()

    X_topk_q2 = X[top_features_q2]
    test_X_topk_q2 = test_X[top_features_q2]

    q2_pred_train_model = LGBMClassifier(**common_params)
    q2_pred_train_model.fit(X_topk_q2, train_df["Q2"])
    q2_pred_train = q2_pred_train_model.predict_proba(X_topk_q2)[:, 1]

    q2_pred_test_cv = cross_val_predict_average(
        X_topk_q2, train_df["Q2"], test_X_topk_q2, lambda: LGBMClassifier(**common_params)
    )

    model_s1 = LGBMClassifier(**common_params, objective="multiclass", num_class=3)
    model_s1.fit(X, train_df[target_multiclass])

    feature_importance = (
        pd.DataFrame({"feature": X.columns, "importance": model_s1.feature_importances_})
        .sort_values(by="importance", ascending=False)
    )
    top_features = feature_importance.head(30)["feature"].tolist()

    X_topk = X[top_features].copy()
    test_X_topk = test_X[top_features].copy()

    X_topk["q2_proba"] = q2_pred_train
    test_X_topk["q2_proba"] = q2_pred_test_cv

    binary_preds = {}
    for col in targets_binary:
        print(f" 학습 중: {col}")
        if col == "Q2":
            proba = q2_pred_test_cv
        else:
            y = train_df[col]
            proba = cross_val_predict_average(X_topk, y, test_X_topk, get_voting_model_logistic)
        binary_preds[col] = (proba > 0.5).astype(int)
        print(f" 완료: {col}")

    y_multi = train_df[target_multiclass]
    s1_class_counts = y_multi.value_counts(normalize=True).sort_index()
    s1_class_weights = 1 / s1_class_counts
    s1_sample_weights = y_multi.map(s1_class_weights)

    model_s1_weighted = LGBMClassifier(**common_params, objective="multiclass", num_class=3)
    model_s1_weighted.fit(X_topk, y_multi, sample_weight=s1_sample_weights)
    multiclass_pred = model_s1_weighted.predict(test_X_topk)

    print(" 모든 타깃 학습 및 예측 완료!")
    return binary_preds, multiclass_pred
