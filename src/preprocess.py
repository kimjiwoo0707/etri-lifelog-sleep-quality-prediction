import pandas as pd
import numpy as np
import ast
from functools import reduce


def get_time_block(hour: int) -> str:
    if 0 <= hour < 6:
        return "early_morning"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def process_mACStatus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    results = []

    for (subj, date), group in df.groupby(["subject_id", "date"]):
        status = group["m_charging"].values
        ratio_charging = status.mean()
        transitions = (status[1:] != status[:-1]).sum()

        lengths, cur = [], 0
        for val in status:
            if val == 1:
                cur += 1
            elif cur > 0:
                lengths.append(cur)
                cur = 0
        if cur > 0:
            lengths.append(cur)

        results.append({
            "subject_id": subj,
            "date": date,
            "charging_ratio": ratio_charging,
            "charging_transitions": transitions,
            "avg_charging_duration": np.mean(lengths) if lengths else 0,
            "max_charging_duration": np.max(lengths) if lengths else 0,
        })

    return pd.DataFrame(results)


def process_mActivity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    summary = []

    for (subj, date), group in df.groupby(["subject_id", "date"]):
        counts = group["m_activity"].value_counts(normalize=True)
        row = {"subject_id": subj, "date": date}

        for i in range(9):
            row[f"activity_{i}_ratio"] = counts.get(i, 0)

        row["dominant_activity"] = group["m_activity"].mode()[0]
        row["num_unique_activities"] = group["m_activity"].nunique()
        summary.append(row)

    return pd.DataFrame(summary)


def process_mBle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    features = []

    for _, row in df.iterrows():
        entry = ast.literal_eval(row["m_ble"]) if isinstance(row["m_ble"], str) else row["m_ble"]
        rssi_list = []
        class_0_cnt = class_other_cnt = 0

        for device in entry:
            try:
                rssi = int(device["rssi"])
                rssi_list.append(rssi)
                if str(device["device_class"]) == "0":
                    class_0_cnt += 1
                else:
                    class_other_cnt += 1
            except:
                continue

        features.append({
            "subject_id": row["subject_id"],
            "date": row["date"],
            "device_class_0_cnt": class_0_cnt,
            "device_class_others_cnt": class_other_cnt,
            "rssi_mean": np.mean(rssi_list) if rssi_list else np.nan,
            "rssi_min": np.min(rssi_list) if rssi_list else np.nan,
            "rssi_max": np.max(rssi_list) if rssi_list else np.nan,
        })

    grouped = pd.DataFrame(features).groupby(["subject_id", "date"]).agg({
        "device_class_0_cnt": "sum",
        "device_class_others_cnt": "sum",
        "rssi_mean": "mean",
        "rssi_min": "min",
        "rssi_max": "max",
    }).reset_index()

    total_cnt = grouped["device_class_0_cnt"] + grouped["device_class_others_cnt"]
    grouped["device_class_0_ratio"] = grouped["device_class_0_cnt"] / total_cnt.replace(0, np.nan)
    grouped["device_class_others_ratio"] = grouped["device_class_others_cnt"] / total_cnt.replace(0, np.nan)
    grouped.drop(columns=["device_class_0_cnt", "device_class_others_cnt"], inplace=True)
    return grouped


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_outdoor_like_alt"] = (
        (df["step_sum"] > 1500)
        | ((df["screen_on_ratio"] > 0.3) & (df["light_day_mean"] > 250))
    ).astype(int)

    df["is_active_user"] = (df["step_sum"] > 5000).astype(int)
    df["screen_heavy_user"] = (df["screen_on_ratio"] > 0.5).astype(int)

    df["night_light_exposure"] = (
        (df["light_night_mean"] > 100) & (df["light_night_ratio"] > 0.4)
    ).astype(int)
    return df


def merge_all_features(feature_dfs: list) -> pd.DataFrame:
    merged = reduce(
        lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"),
        feature_dfs,
    )
    merged = sanitize_column_names(merged)
    merged = add_behavioral_features(merged)
    return merged
