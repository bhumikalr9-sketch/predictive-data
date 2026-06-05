import pandas as pd
import numpy as np


def preprocess_data(df):
    """
    Basic preprocessing
    """

    df = df.copy()

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date")

    return df


def create_rolling_features(df, window=10):

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

    for col in metrics:

        df[f"{col}_rolling_mean"] = (
            df[col]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        df[f"{col}_rolling_std"] = (
            df[col]
            .rolling(window=window, min_periods=1)
            .std()
        )

    return df


def create_time_features(df):

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()

    return df


def feature_engineering_pipeline(df):

    df = preprocess_data(df)

    df = create_rolling_features(df)

    df = create_time_features(df)

    return df
