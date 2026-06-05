import pandas as pd


def get_dataset_summary(df):

    return {

        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Devices": df["device"].nunique(),
        "Failures": int(df["failure"].sum())
    }


def get_device_list(df):

    return sorted(
        df["device"].unique()
    )


def filter_device(df, device):

    return df[
        df["device"] == device
    ]


def calculate_failure_rate(df):

    total = len(df)

    failures = df["failure"].sum()

    if total == 0:
        return 0

    return round(
        (failures / total) * 100,
        2
    )


def get_latest_status(df):

    latest = df.sort_values(
        "date"
    ).iloc[-1]

    return {

        "Device": latest["device"],
        "Failure": latest["failure"],
        "Metric1": latest["metric1"],
        "Metric2": latest["metric2"],
        "Metric3": latest["metric3"]
    }


def create_download_csv(df):

    return df.to_csv(
        index=False
    ).encode("utf-8")
