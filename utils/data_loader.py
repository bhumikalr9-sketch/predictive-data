def load_and_process_data(uploaded_file):
    import pandas as pd

    df = pd.read_csv(uploaded_file)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

    return df
