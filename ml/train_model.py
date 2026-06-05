import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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


def prepare_data(df):

    X = df[FEATURES]

    y = df["failure"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_all_models(df):

    X_train, X_test, y_train, y_test = prepare_data(df)

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "KNN":
        KNeighborsClassifier(n_neighbors=5),

        "Decision Tree":
        DecisionTreeClassifier(random_state=42),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return (
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test
    )


def save_model(model, filename):

    joblib.dump(model, filename)


def load_model(filename):

    return joblib.load(filename)
