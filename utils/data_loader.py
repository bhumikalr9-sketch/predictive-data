# utils/data_loader.py

import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = [
    "Timestamp",
    "Equipment_ID",
    "Temperature (°C)",
    "Vibration (m/s²)",
    "Voltage (V)",
    "Failure_Status"
]


def load_data(uploaded_file):
    """
    Load CSV dataset.
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def validate_data(df):
    """
    Validate required columns.
    """
    if df is None:
        return False

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"Missing required columns: {', '.join(missing_columns)}"
        )
        return False

    return True


def get_dataset_summary(df):
    """
    Dataset summary.
    """
    if df is None:
        return {}

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }


def get_numeric_columns(df):
    """
    Get numerical columns.
    """
    if df is None:
        return []

    return df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


def clean_data(df):
    """
    Basic cleaning.
    """
    if df is None:
        return None

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def convert_timestamp(df):
    """
    Convert timestamp column.
    """
    if df is None:
        return None

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

    return df


def load_and_process_data(uploaded_file):
    """
    Complete pipeline.
    """
    df = load_data(uploaded_file)

    if df is None:
        return None

    df = convert_timestamp(df)
    df = clean_data(df)

    return df
