import os
import glob
import pandas as pd
import numpy as np


def load_raw_data(data_dir: str):
    parquet_files = glob.glob(os.path.join(data_dir, 'ch2025_*.parquet'))
    lifelog_data = {}

    for file_path in parquet_files:
        name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
        lifelog_data[name] = pd.read_parquet(file_path)
        print(f"Loaded: {name}, shape = {lifelog_data[name].shape}")

    return lifelog_data


def load_csvs(data_dir: str):
    metrics_train = pd.read_csv(os.path.join(data_dir, 'ch2025_metrics_train.csv'))
    sample_submission = pd.read_csv(os.path.join(data_dir, 'ch2025_submission_sample.csv'))
    return metrics_train, sample_submission


def make_test_keys(sample_submission: pd.DataFrame):
    sample_submission['lifelog_date'] = pd.to_datetime(sample_submission['lifelog_date'])
    return set(zip(sample_submission['subject_id'], sample_submission['lifelog_date'].dt.date))


def split_test_train(df, test_keys, subject_col='subject_id', timestamp_col='timestamp'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])
    df['date_only'] = df[timestamp_col].dt.date
    df['key'] = list(zip(df[subject_col], df['date_only']))

    test_df = df[df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])
    train_df = df[~df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])

    print(f"Split → train: {train_df.shape}, test: {test_df.shape}")
    return test_df, train_df


def basic_feature_extraction(df: pd.DataFrame):
    if 'time' not in df.columns:
        if 'timestamp' in df.columns:
            df['time'] = df['timestamp']
        elif 'date' in df.columns:
            df['time'] = pd.to_datetime(df['date'])

    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day'] = df['time'].dt.day
    df['weekday'] = df['time'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['time_timestamp'] = df['time'].apply(lambda x: x.timestamp())

    return df.drop(columns=['time'])
