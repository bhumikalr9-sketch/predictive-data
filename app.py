import streamlit as st
import pandas as pd
import os

from utils.data_loader import (
    load_data,
    convert_timestamp,
    get_equipment_list,
    filter_equipment,
    get_dataset_info
)

from utils.preprocessing import (
    full_preprocessing
)

from utils.helper import (
    calculate_operating_hours,
    current_temperature,
    count_alerts
)

from analytics.feature_engineering import (
    create_all_features
)

from analytics.anomaly_detection import (
    detect_anomalies,
    get_anomaly_records
)

from analytics.root_cause_analysis import (
    failure_summary,
    root_cause_records
)

from analytics.visualization import (
    plot_temperature,
    plot_vibration,
    plot_failure_distribution,
    plot_correlation_heatmap,
    create_wordcloud,
    plot_anomaly_trend
)

from ml.train_models import (
    train_and_save_models
)

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance Dashboard")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)

threshold = st.sidebar.slider(
    "Anomaly Threshold (Z-Score)",
    min_value=2.0,
    max_value=4.0,
    value=3.0,
    step=0.1
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

if uploaded_file:

    df = load_data(uploaded_file)

    df = convert_timestamp(df)

    df = full_preprocessing(df)

    df = create_all_features(df)

    # ----------------------------------------
    # Equipment Filter
    # ----------------------------------------

    equipment_list = get_equipment_list(df)

    selected_equipment = st.sidebar.selectbox(
        "Equipment ID",
        ["All"] + list(equipment_list)
    )

    df = filter_equipment(
        df,
        selected_equipment
    )

    # ----------------------------------------
    # Anomaly Detection
    # ----------------------------------------

    df = detect_anomalies(
        df,
        threshold
    )

    # ----------------------------------------
    # Dataset Info
    # ----------------------------------------

    st.subheader("Dataset Information")

    info = get_dataset_info(df)

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", info["Rows"])
    c2.metric("Columns", info["Columns"])
    c3.metric("Missing Values", info["Missing Values"])

    # ----------------------------------------
    # KPI Metrics
    # ----------------------------------------

    st.subheader("Real-Time Metrics")

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "Total Operating Hours",
        calculate_operating_hours(df)
    )

    k2.metric(
        "Current Temperature",
        current_temperature(df)
    )

    k3.metric(
        "Active Critical Alerts",
        count_alerts(df)
    )

    # ----------------------------------------
    # Sensor Data
    # ----------------------------------------

    st.subheader("Sensor Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ----------------------------------------
    # Temperature Chart
    # ----------------------------------------

    st.subheader("Temperature Trend")

    fig_temp = plot_temperature(df)

    st.pyplot(fig_temp)

    # ----------------------------------------
    # Vibration Chart
    # ----------------------------------------

    st.subheader("Vibration Trend")

    fig_vib = plot_vibration(df)

    st.pyplot(fig_vib)

    # ----------------------------------------
    # Anomaly Trend
    # ----------------------------------------

    st.subheader("Anomaly Detection Trend")

    fig_anomaly = plot_anomaly_trend(df)

    st.pyplot(fig_anomaly)

    # ----------------------------------------
    # Anomaly Records
    # ----------------------------------------

    st.subheader("Detected Anomalies")

    anomaly_df = get_anomaly_records(df)

    st.dataframe(
        anomaly_df,
        use_container_width=True
    )

    # Save anomaly report

    os.makedirs(
        "reports",
        exist_ok=True
    )

    anomaly_df.to_csv(
        "reports/anomaly_report.csv",
        index=False
    )

    # ----------------------------------------
    # Failure Summary
    # ----------------------------------------

    st.subheader("Failure Summary")

    summary = failure_summary(df)

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "Total Failures",
        summary["Total Failures"]
    )

    f2.metric(
        "Average Temperature",
        summary["Average Temperature"]
    )

    f3.metric(
        "Average Vibration",
        summary["Average Vibration"]
    )

    f4.metric(
        "Average Voltage",
        summary["Average Voltage"]
    )

    # ----------------------------------------
    # Root Cause Analysis
    # ----------------------------------------

    st.subheader("Root Cause Analysis")

    root_df = root_cause_records(df)

    st.dataframe(
        root_df,
        use_container_width=True
    )

    # ----------------------------------------
    # Failure Distribution
    # ----------------------------------------

    st.subheader("Failure Distribution")

    fig_fail = plot_failure_distribution(df)

    st.pyplot(fig_fail)

    # ----------------------------------------
    # Correlation Heatmap
    # ----------------------------------------

    st.subheader("Correlation Heatmap")

    fig_heat = plot_correlation_heatmap(df)

    st.pyplot(fig_heat)

    # ----------------------------------------
    # Word Cloud
    # ----------------------------------------

    st.subheader("Failure Type Word Cloud")

    wc = create_wordcloud(df)

    if wc:
        st.pyplot(wc)

    # ----------------------------------------
    # Machine Learning Models
    # ----------------------------------------

    st.subheader("Model Evaluation")

    try:

        results_df = train_and_save_models(df)

        st.dataframe(
            results_df,
            use_container_width=True
        )

        results_df.to_csv(
            "reports/model_evaluation.csv",
            index=False
        )

        st.success(
            "Models trained successfully."
        )

    except Exception as e:

        st.error(
            f"Training Error: {e}"
        )

    # ----------------------------------------
    # Downloads
    # ----------------------------------------

    st.subheader("Download Reports")

    if os.path.exists(
        "reports/anomaly_report.csv"
    ):

        with open(
            "reports/anomaly_report.csv",
            "rb"
        ) as f:

            st.download_button(
                "Download Anomaly Report",
                f,
                file_name="anomaly_report.csv"
            )

    if os.path.exists(
        "reports/model_evaluation.csv"
    ):

        with open(
            "reports/model_evaluation.csv",
            "rb"
        ) as f:

            st.download_button(
                "Download Model Evaluation",
                f,
                file_name="model_evaluation.csv"
            )

else:

    st.info(
        "Upload sensor_maintenance_data.csv to begin analysis."
    )
