from src.preprocess import load_raw_data, load_csvs, make_test_keys
from src.feature_engineering import (
    process_mACStatus,
    process_mActivity,
    process_mBle,
    merge_all_features,
)
from src.train import train_all_targets
from src.inference import generate_submission


def main():
    print(" ETRI Lifelog AI Pipeline Started")

    base_dir = "/content/drive/MyDrive/ETRI/ETRI_lifelog_dataset"
    data_dir = f"{base_dir}/ch2025_data_items"

    lifelog_data = load_raw_data(data_dir)
    metrics_train, sample_submission = load_csvs(base_dir)
    test_keys = make_test_keys(sample_submission)

    print(" Feature engineering started...")
    mAC_df = process_mACStatus(lifelog_data["mACStatus"])
    mAct_df = process_mActivity(lifelog_data["mActivity"])
    mBle_df = process_mBle(lifelog_data["mBle"])

    merged_df = merge_all_features([mAC_df, mAct_df, mBle_df])
    print(f" Merged feature shape: {merged_df.shape}")

    metrics_train["lifelog_date"] = pd.to_datetime(metrics_train["lifelog_date"]).dt.date
    merged_df["date"] = pd.to_datetime(merged_df["date"]).dt.date

    train_df = pd.merge(
        metrics_train.rename(columns={"lifelog_date": "date"}),
        merged_df,
        on=["subject_id", "date"],
        how="inner",
    )

    merged_keys = merged_df[["subject_id", "date"]]
    train_keys = metrics_train[["subject_id", "lifelog_date"]].rename(columns={"lifelog_date": "date"})
    test_keys_df = pd.merge(merged_keys, train_keys, on=["subject_id", "date"], how="left", indicator=True)
    test_keys_df = test_keys_df[test_keys_df["_merge"] == "left_only"].drop(columns=["_merge"])
    test_df = pd.merge(test_keys_df, merged_df, on=["subject_id", "date"], how="left")


    X = train_df.drop(columns=["subject_id", "sleep_date", "date", "Q1", "Q2", "Q3", "S1", "S2", "S3"])
    X.fillna(0, inplace=True)
    test_X = test_df.drop(columns=["subject_id", "date"])
    test_X.fillna(0, inplace=True)

    print(" Training models...")
    binary_preds, multiclass_pred = train_all_targets(X, train_df, test_X)

    print(" Generating submission file...")
    submission = generate_submission(
        sample_submission,
        multiclass_pred,
        binary_preds,
        binary_preds["S2"],
        binary_preds["S3"],
    )

    print(" Pipeline finished successfully!")

if __name__ == "__main__":
    main()
