import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_correlation_heatmap(corr_matrix):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    plt.title(
        "Correlation Heatmap"
    )

    return fig


def plot_feature_importance(
        importance_df):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax
    )

    plt.title(
        "Feature Importance"
    )

    return fig


def plot_failure_distribution(df):

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.countplot(
        x="failure",
        data=df,
        ax=ax
    )

    plt.title(
        "Failure Distribution"
    )

    return fig


def generate_wordcloud(
        importance_df):

    words = []

    for _, row in importance_df.iterrows():

        repeat = int(
            row["Importance"] * 1000
        )

        words.extend(
            [row["Feature"]] * repeat
        )

    text = " ".join(words)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    plt.title(
        "Root Cause Word Cloud"
    )

    return fig


def plot_sensor_trend(
        df,
        metric="metric1"):

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["date"],
        df[metric]
    )

    ax.set_title(
        f"{metric} Trend"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(metric)

    return fig
