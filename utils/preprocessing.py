import pandas as pd
import numpy as np


def clean_data(df):

    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df = df.fillna(
        df.select_dtypes(
            include=np.number
        ).mean()
    )

    return df


def convert_datetime(df):

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def sort_data(df):

    return df.sort_values(
        by="date"
    )


def preprocess_pipeline(df):

    df = clean_data(df)

    df = convert_datetime(df)

    df = sort_data(df)

    return df
