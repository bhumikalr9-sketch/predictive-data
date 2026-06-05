import pandas as pd


FEATURES = [
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


def predict_failure(model, input_data):

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    prediction = model.predict(
        input_df
    )[0]

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(
            input_df
        )[0][1]

    return prediction, probability


def prediction_label(prediction):

    if prediction == 1:
        return "Failure Risk Detected"

    return "Machine Healthy"
