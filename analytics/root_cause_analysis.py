import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def feature_importance_analysis(df):

    features = [
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

    X = df[features]

    y = df["failure"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    return importance_df


def correlation_analysis(df):

    cols = [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9",
        "failure"
    ]

    return df[cols].corr()


def failure_summary(df):

    total_records = len(df)

    total_failures = df["failure"].sum()

    failure_rate = (
        total_failures / total_records
    ) * 100

    return {
        "Total Records": total_records,
        "Total Failures": int(total_failures),
        "Failure Rate": round(failure_rate, 2)
    }
