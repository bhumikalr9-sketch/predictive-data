import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

st.title("🔧 Predictive Maintenance System")
st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Upload Sensor Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

threshold = st.sidebar.slider(
    "Z-Score Threshold",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

# -----------------------------
# LOAD DATA
# -----------------------------

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset Loaded Successfully")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------------
        # DATETIME CONVERSION
        # -----------------------------------

        timestamp_col = None

        for col in df.columns:
            if "time" in col.lower():
                timestamp_col = col
                break

        if timestamp_col:

            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col],
                errors="coerce"
            )

            df = df.dropna(subset=[timestamp_col])

            df.set_index(timestamp_col, inplace=True)

        # -----------------------------------
        # EQUIPMENT FILTER
        # -----------------------------------

        equipment_col = None

        for col in df.columns:
            if "equipment" in col.lower():
                equipment_col = col
                break

        if equipment_col:

            equipment_ids = df[equipment_col].unique()

            selected_equipment = st.sidebar.selectbox(
                "Select Equipment",
                equipment_ids
            )

            df = df[
                df[equipment_col] == selected_equipment
            ]

        # -----------------------------------
        # NUMERIC COLUMNS
        # -----------------------------------

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) == 0:
            st.error("No numeric columns found.")
            st.stop()

        # -----------------------------------
        # METRICS
        # -----------------------------------

        col1, col2, col3 = st.columns(3)

        operating_hours = len(df)

        current_temp = "N/A"

        temp_cols = [
            c for c in df.columns
            if "temp" in c.lower()
        ]

        if len(temp_cols) > 0:
            current_temp = round(
                df[temp_cols[0]].iloc[-1],
                2
            )

        # -----------------------------------
        # ANOMALY DETECTION
        # -----------------------------------

        anomaly_count = 0

        if len(numeric_cols) > 0:

            z_scores = np.abs(
                (
                    df[numeric_cols] -
                    df[numeric_cols].mean()
                )
                /
                df[numeric_cols].std()
            )

            df["Anomaly_Alert"] = (
                z_scores > threshold
            ).any(axis=1).astype(int)

            anomaly_count = int(
                df["Anomaly_Alert"].sum()
            )

        col1.metric(
            "Operating Records",
            operating_hours
        )

        col2.metric(
            "Current Temperature",
            current_temp
        )

        col3.metric(
            "Critical Alerts",
            anomaly_count
        )

        st.markdown("---")

        # -----------------------------------
        # SENSOR TREND
        # -----------------------------------

        st.subheader("Sensor Trend Analysis")

        selected_sensor = st.selectbox(
            "Select Sensor",
            numeric_cols
        )

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            df.index,
            df[selected_sensor]
        )

        threshold_line = (
            df[selected_sensor].mean()
            +
            threshold *
            df[selected_sensor].std()
        )

        ax.axhline(
            threshold_line,
            linestyle="--"
        )

        ax.set_title(selected_sensor)

        st.pyplot(fig)

        # -----------------------------------
        # ROLLING ANALYSIS
        # -----------------------------------

        st.subheader("Rolling Mean Analysis")

        rolling_df = pd.DataFrame()

        rolling_df["Actual"] = df[selected_sensor]

        rolling_df["Rolling Mean"] = (
            df[selected_sensor]
            .rolling(window=10)
            .mean()
        )

        st.line_chart(
            rolling_df
        )

        # -----------------------------------
        # ANOMALY TABLE
        # -----------------------------------

        st.subheader(
            "Detected Anomalies"
        )

        anomalies = df[
            df["Anomaly_Alert"] == 1
        ]

        st.dataframe(anomalies)

        # -----------------------------------
        # ROOT CAUSE ANALYSIS
        # -----------------------------------

        st.subheader(
            "Root Cause Analysis"
        )

        correlation = df[
            numeric_cols
        ].corr()

        st.dataframe(
            correlation
        )

        # -----------------------------------
        # WORD CLOUD
        # -----------------------------------

        st.subheader(
            "Feature Importance Word Cloud"
        )

        text = " ".join(
            numeric_cols
        )

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        fig2, ax2 = plt.subplots()

        ax2.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax2.axis("off")

        st.pyplot(fig2)

        # -----------------------------------
        # MACHINE LEARNING
        # -----------------------------------

        st.subheader(
            "Predictive Maintenance Models"
        )

        target_col = None

        possible_targets = [
            c for c in df.columns
            if (
                "failure" in c.lower()
                or
                "fault" in c.lower()
                or
                "status" in c.lower()
            )
        ]

        if len(possible_targets) > 0:
            target_col = possible_targets[0]

        if target_col:

            X = df[numeric_cols].copy()

            if target_col in X.columns:
                X = X.drop(
                    columns=[target_col]
                )

            y = df[target_col]

            X = X.fillna(
                X.mean()
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            models = {
                "Linear Regression":
                    LinearRegression(),

                "KNN Regression":
                    KNeighborsRegressor(),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )
            }

            results = []

            for name, model in models.items():

                try:

                    model.fit(
                        X_train,
                        y_train
                    )

                    pred = model.predict(
                        X_test
                    )

                    mse = mean_squared_error(
                        y_test,
                        pred
                    )

                    rmse = np.sqrt(mse)

                    r2 = r2_score(
                        y_test,
                        pred
                    )

                    results.append(
                        [
                            name,
                            round(mse, 4),
                            round(rmse, 4),
                            round(r2, 4)
                        ]
                    )

                except:
                    pass

            result_df = pd.DataFrame(
                results,
                columns=[
                    "Model",
                    "MSE",
                    "RMSE",
                    "R2"
                ]
            )

            st.dataframe(
                result_df
            )

        else:

            st.warning(
                "No Failure/Fault column found for ML prediction."
            )

    except Exception as e:

        st.error(
            f"Error Loading File: {e}"
        )

else:

    st.info(
        "Upload a CSV file to begin analysis."
    )
