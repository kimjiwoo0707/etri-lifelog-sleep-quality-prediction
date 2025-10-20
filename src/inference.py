import pandas as pd
import numpy as np


def generate_submission(
    sample_submission: pd.DataFrame,
    multiclass_pred: np.ndarray,
    binary_preds: dict,
    s2_pred: np.ndarray,
    s3_pred: np.ndarray,
    output_path: str = "submission_final.csv"
):

    submission = sample_submission[["subject_id", "sleep_date", "lifelog_date"]].copy()
    submission["lifelog_date"] = pd.to_datetime(submission["lifelog_date"]).dt.date

    submission["S1"] = multiclass_pred
    submission["Q1"] = binary_preds["Q1"].astype(int)
    submission["Q2"] = binary_preds["Q2"].astype(int)
    submission["Q3"] = binary_preds["Q3"].astype(int)
    submission["S2"] = s2_pred.astype(int)
    submission["S3"] = s3_pred.astype(int)

    submission = submission[
        ["subject_id", "sleep_date", "lifelog_date", "Q1", "Q2", "Q3", "S1", "S2", "S3"]
    ]

    submission.to_csv(output_path, index=False)
    print(f" Saved: {output_path} ({len(submission)} rows)")
    return submission


def compare_submissions(csv_path_1: str, csv_path_2: str):

    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    df1 = df1.sort_values(by=["subject_id", "lifelog_date"]).reset_index(drop=True)
    df2 = df2.sort_values(by=["subject_id", "lifelog_date"]).reset_index(drop=True)

    assert df1.shape == df2.shape, " 두 파일의 shape가 다릅니다!"

    diff_mask = df1 != df2
    diff_rows = diff_mask.any(axis=1)
    diff_count = diff_rows.sum()

    print(f" 총 {diff_count}개의 row가 다릅니다.")
    if diff_count > 0:
        diff_detail = df1[diff_rows].copy()
        diff_detail["diff_columns"] = diff_mask[diff_rows].apply(
            lambda row: df1.columns[row].tolist(), axis=1
        )
        print(diff_detail[["subject_id", "lifelog_date", "diff_columns"]].head())

    return diff_count
