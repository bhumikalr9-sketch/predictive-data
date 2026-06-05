# utils/data_loader.py

import pandas as pd


def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def validate_data(df):
    return True


def get_dataset_summary(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum())
    }


def clean_data(df):
    return df.drop_duplicates()


def get_numeric_columns(df):
    return df.select_dtypes(include=["number"]).columns.tolist()


def convert_timestamp(df):
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )
    return df


def load_and_process_data(uploaded_file):
    df = load_data(uploaded_file)
    df = convert_timestamp(df)
    df = clean_data(df)
    return df
