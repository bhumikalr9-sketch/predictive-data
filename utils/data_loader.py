import pandas as pd


def load_data(uploaded_file):
    """
    Load CSV dataset
    """

    try:

        df = pd.read_csv(uploaded_file)

        return df

    except Exception as e:

        raise Exception(
            f"Error loading file: {e}"
        )


def validate_columns(df):

    required_columns = [

        "date",
        "device",
        "failure",

        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9"
    ]

    missing_columns = [

        col
        for col in required_columns
        if col not in df.columns
    ]

    return missing_columns
