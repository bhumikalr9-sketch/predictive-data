import numpy as np
import pandas as pd


def calculate_zscore(series):

    mean = series.mean()
    std = series.std()

    zscore = (series - mean) / std

    return zscore


def detect_anomalies(df, threshold=3.0):

    df = df.copy()

    metrics = [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9",
    ]

    anomaly_flags = []

    for col in metrics:

        z = calculate_zscore(df[col])

        df[f"{col}_zscore"] = z

        anomaly_flags.append(
            np.abs(z) > threshold
        )

    combined = np.column_stack(anomaly_flags)

    df["Anomaly_Alert"] = np.where(
        combined.any(axis=1),
        1,
        0
    )

    return df


def get_anomaly_records(df):

    return df[
        df["Anomaly_Alert"] == 1
    ]
