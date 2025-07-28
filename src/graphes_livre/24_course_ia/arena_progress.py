import pickle
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem, hf_hub_download
from scipy.special import expit

from graphes_livre import apply_template, get_output_path

USE_CONTINENTS = True


def download_latest_data_from_space(
    repo_id: str, file_type: Literal["pkl", "csv"]
) -> str:
    """
    Downloads the latest data file of the specified file type from the given repository space.
    Args:
        repo_id (str): The ID of the repository space.
        file_type (Literal["pkl", "csv"]): The type of the data file to download. Must be either "pkl" or "csv".
    Returns:
        str: The local file path of the downloaded data file.
    """

    def extract_date(filename):
        return filename.split("/")[-1].split(".")[0].split("_")[-1]

    fs = HfFileSystem()
    data_file_path = f"spaces/{repo_id}/*.{file_type}"
    files = fs.glob(data_file_path)
    files = [
        file for file in files if "leaderboard_table" in file or "elo_results" in file
    ]
    latest_file = sorted(files, key=extract_date, reverse=True)[0]

    latest_filepath_local = hf_hub_download(
        repo_id=repo_id,
        filename=latest_file.split("/")[-1],
        repo_type="space",
    )
    print(latest_file.split("/")[-1])
    return latest_filepath_local


def get_constants(dfs):
    """
    Calculate and return the minimum and maximum Elo scores, as well as the maximum number of models per month.
    Parameters:
    - dfs (dict): A dictionary containing DataFrames for different categories.
    Returns:
    - min_elo_score (float): The minimum Elo score across all DataFrames.
    - max_elo_score (float): The maximum Elo score across all DataFrames.
    """
    filter_ranges = {}
    for k, df in dfs.items():
        filter_ranges[k] = {
            "min_elo_score": df["rating"].min().round(),
            "max_elo_score": df["rating"].max().round(),
        }

    min_elo_score = float("inf")
    max_elo_score = float("-inf")

    for _, value in filter_ranges.items():
        min_elo_score = min(min_elo_score, value["min_elo_score"])
        max_elo_score = max(max_elo_score, value["max_elo_score"])

    return min_elo_score, max_elo_score


def format_data(df):
    """
    Formats the given DataFrame by performing the following operations:
    - Converts the 'License' column values to 'Proprietary LLM' if they are in PROPRIETARY_LICENSES, otherwise 'Open LLM'.
    - Converts the 'Release Date' column to datetime format.
    - Adds a new 'Month-Year' column by extracting the month and year from the 'Release Date' column.
    - Rounds the 'rating' column to the nearest integer.
    - Resets the index of the DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to be formatted.
    Returns:
        pandas.DataFrame: The formatted DataFrame.
    """

    PROPRIETARY_LICENSES = ["Proprietary", "Proprietory"]

    df["License"] = df["License"].apply(
        lambda x: "Proprietary LLM" if x in PROPRIETARY_LICENSES else "Open LLM"
    )
    df["Release Date"] = pd.to_datetime(df["Release Date"])
    df["Month-Year"] = df["Release Date"].dt.to_period("M")
    df["rating"] = df["rating"].round()
    return df.reset_index(drop=True)


# Function to create sigmoid transition
def sigmoid_transition(x, x0, k=0.1):
    return expit(k * (x - x0))


def linear_transition(x, x0, k=0.1):
    return x0 + k * (x - x0)


def main():
    KEY_TO_CATEGORY_NAME = {
        "full": "Overall",
        "dedup": "De-duplicate Top Redundant Queries (soon to be default)",
        "math": "Math",
        "if": "Instruction Following",
        "multiturn": "Multi-Turn",
        "coding": "Coding",
        "hard_6": "Hard Prompts (Overall)",
        "hard_english_6": "Hard Prompts (English)",
        "long_user": "Longer Query",
        "english": "English",
        "chinese": "Chinese",
        "french": "French",
        "german": "German",
        "spanish": "Spanish",
        "russian": "Russian",
        "japanese": "Japanese",
        "korean": "Korean",
        "no_tie": "Exclude Ties",
        "no_short": "Exclude Short Query (< 5 tokens)",
        "no_refusal": "Exclude Refusal",
        "overall_limit_5_user_vote": "overall_limit_5_user_vote",
        "full_old": "Overall (Deprecated)",
    }

    # gather ELO data
    latest_elo_file_local = download_latest_data_from_space(
        repo_id="lmsys/chatbot-arena-leaderboard", file_type="pkl"
    )

    with open(latest_elo_file_local, "rb") as fin:
        # Monkey patch plotly to handle deprecated 'heatmapgl' property
        import plotly.graph_objs.layout.template._data as template_data

        original_init = template_data.Data.__init__

        def patched_init(self, arg=None, **kwargs):
            # Remove deprecated properties before calling original init
            if isinstance(arg, dict) and "heatmapgl" in arg:
                arg = dict(arg)  # Create a copy
                del arg["heatmapgl"]  # Remove the problematic property
            if "heatmapgl" in kwargs:
                del kwargs["heatmapgl"]
            return original_init(self, arg, **kwargs)

        template_data.Data.__init__ = patched_init

        elo_results = pickle.load(fin)

        # Restore original method
        template_data.Data.__init__ = original_init

    # TO-DO: need to also include vision
    elo_results = elo_results["text"]

    arena_dfs = {}
    for k in KEY_TO_CATEGORY_NAME.keys():
        if k not in elo_results:
            continue
        arena_dfs[KEY_TO_CATEGORY_NAME[k]] = elo_results[k]["leaderboard_table_df"]

    # gather open llm leaderboard data
    latest_leaderboard_file_local = download_latest_data_from_space(
        repo_id="lmsys/chatbot-arena-leaderboard", file_type="csv"
    )
    leaderboard_df = pd.read_csv(latest_leaderboard_file_local)

    # load release date mapping data
    release_date_mapping = pd.read_json(
        "data/release_date_mapping.json", orient="records"
    )

    # merge leaderboard data with ELO data
    merged_dfs = {}
    for k, v in arena_dfs.items():
        merged_dfs[k] = (
            pd.merge(arena_dfs[k], leaderboard_df, left_index=True, right_on="key")
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )

    # add release dates into the merged data
    for k, v in merged_dfs.items():
        merged_dfs[k] = pd.merge(
            merged_dfs[k],
            release_date_mapping[["key", "Release Date"]],
            on="key",
            how="left",
        )

    # format dataframes
    merged_dfs = {k: format_data(v) for k, v in merged_dfs.items()}

    # get constants
    df = merged_dfs["Overall"]

    top_orgs = df.groupby("Organization")["rating"].max().nlargest(13).index.tolist()
    print(top_orgs)
    top_orgs = [
        el
        for el in top_orgs
        if el not in ["NexusFlow", "Princeton", "Nvidia", "MiniMax", "Zhipu"]
    ]
<<<<<<< HEAD
=======

>>>>>>> 7542881afc3ea8af157b8a7f81fb6b0980037bc7
    df = df.loc[(df["Organization"].isin(top_orgs)) & (df["rating"] > 1000)]
    if USE_CONTINENTS:
        df["Organization"] = df["Organization"].map(
            {
                "OpenAI": "Etats-Unis",
                "Google": "Etats-Unis",
                "xAI": "Etats-Unis",
                "Anthropic": "Etats-Unis",
                "Meta": "Etats-Unis",
                "Alibaba": "Chine",
                "DeepSeek": "Chine",
                "01 AI": "Chine",
                "DeepSeek AI": "Chine",
                "Mistral": "France",
                "Reka AI": "Etats-Unis",
                "Zhipu AI": "Chine",
                "MiniMax": "Chine",
                "Qwen": "Chine",
                "Tencent": "Chine",
                "Moonshot": "Chine",
            }
        )

    print("Missing release dates:")
    print(df.loc[df["Release Date"].isna()][["rating", "key", "Model"]])

    df = df.loc[~df["Release Date"].isna()]

    # Convert Release Date to datetime if it's not already
    df["Release Date"] = pd.to_datetime(df["Release Date"])

    # Sort the DataFrame by Release Date and rating (descending)
    df = df.sort_values(["Release Date", "rating"], ascending=[True, False])
    df = df.loc[
        ~df["Model"]
        .str.lower()
        .apply(lambda x: "early" in x or "preview" in x or "experimental" in x)
    ]

    # Define the current date
    current_date = pd.Timestamp(year=2025, month=6, day=1)

    # Define organization to country mapping and colors
    org_info = {
        "OpenAI": ("#00A67E", "🇺🇸"),  # Teal
        "Google": ("#4285F4", "🇺🇸"),  # Google Blue
        "xAI": ("black", "🇺🇸"),  # Bright Orange
        "Anthropic": ("#cc785c", "🇺🇸"),  # Brown (as requested)
        "Meta": ("#0064E0", "🇺🇸"),  # Facebook Blue
        "Alibaba": ("#6958cf", "🇨🇳"),
        "DeepSeek": ("#C70039", "🇨🇳"),
        "01 AI": ("#11871e", "🇨🇳"),  # Bright Green
        "DeepSeek AI": ("#9900CC", "🇨🇳"),  # Purple
        "Mistral": ("#ff7000", "🇫🇷"),  # Mistral Orange (as requested)
        "AI21 Labs": ("#1E90FF", "🇮🇱"),  # Dodger Blue,
        "Reka AI": ("#FFC300", "🇺🇸"),
        "Zhipu AI": ("#FFC300", "🇨🇳"),
        "Moonshot": ("#000000", "🇨🇳"),
        "Qwen": ("#000000", "🇨🇳"),
        "Tencent": ("#BBBBBB", "🇨🇳"),
        "MiniMax": ("#000000", "🇨🇳"),
        "Cohere": ("#d9a6e5", "🇨🇦"),
    }
    if USE_CONTINENTS:
        org_info = {
            "Etats-Unis": ("#4285F4", "🇺🇸"),
            "Chine": ("#C70039", "🇨🇳"),
            "France": ("#ff7000", "🇫🇷"),
        }

    # Create figure
    fig = go.Figure()

    for i, org in enumerate(
        df.groupby("Organization")["rating"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    ):
        org_data = df[df["Organization"] == org]

        if len(org_data) > 0:
            x_values = []
            y_values = []
            current_best = -np.inf
            best_models = []

            # Group by date and get the best model for each date
            daily_best = (
                org_data.sort_values("rating", ascending=False)
                .groupby("Release Date")
                .first()
                .reset_index()
            )

            # Filter out updates less than N days apart, keeping only the later one
            filtered_best = []
            for _, row in daily_best.iterrows():
                if not filtered_best:
                    filtered_best.append(row)
                else:
                    days_diff = (
                        row["Release Date"] - filtered_best[-1]["Release Date"]
                    ).days
                    if days_diff >= 20:
                        filtered_best.append(row)
                    else:
                        # Replace the previous entry with the later one
                        filtered_best[-1] = row

            daily_best = pd.DataFrame(filtered_best)

            for _, row in daily_best.iterrows():
                if row["rating"] > current_best:
                    if len(x_values) > 0:
                        # Create smooth transition
                        transition_days = (row["Release Date"] - x_values[-1]).days
                        transition_points = pd.date_range(
                            x_values[-1],
                            row["Release Date"],
                            periods=max(100, transition_days),
                        )
                        x_values.extend(transition_points)

                        transition_y = current_best + (
                            row["rating"] - current_best
                        ) * np.linspace(0, 1, len(transition_points))
                        # * sigmoid_transition(
                        #     np.linspace(-6, 6, len(transition_points)), 0, k=1
                        # )
                        y_values.extend(transition_y)

                    x_values.append(row["Release Date"])
                    y_values.append(row["rating"])
                    current_best = row["rating"]
                    best_models.append(row)

            # Extend the line to the current date
            if x_values[-1] < current_date:
                x_values.append(current_date)
                y_values.append(current_best)

            # Get org color and flag
            color, flag = org_info.get(org, ("#808080", ""))

            # Add line plot
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=f"{i + 1}. {org} {flag}",
                    line=dict(color=color, width=2),
                    hoverinfo="skip",
                )
            )

            fig.add_annotation(
                x=x_values[-1],
                y=y_values[-1],
                text=f" {org}",
                showarrow=False,
                font=dict(size=17, color=color),
                xref="x",
                yref="y",
                xanchor="left",
            )

            # Add scatter plot for best model points
            # best_models_df = pd.DataFrame(best_models)
            # fig.add_trace(
            #     go.Scatter(
            #         x=best_models_df["Release Date"],
            #         y=best_models_df["rating"],
            #         mode="markers",
            #         name=org,
            #         showlegend=False,
            #         marker=dict(color=color, size=8, symbol="circle"),
            #         text=best_models_df["Model"],
            #         hovertemplate="<b>%{text}</b><br>Date: %{x}<br>ELO Score: %{y:.2f}<extra></extra>",
            #     )
            # )

    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Score ELO",
        # legend_title="Classement en Janvier 24",
        hovermode="closest",
        xaxis_range=[
            pd.Timestamp("2024-01-01"),
            current_date,
        ],  # Extend x-axis for labels
        yaxis_range=[1151, 1500],
        showlegend=False,
        # margin=dict(r=-60),
    )
    apply_template(fig, width=700, height=500)
    fig.update_xaxes(
        tickformat="%m-%Y",
        range=[pd.Timestamp("2024-01-01"), current_date + pd.Timedelta(days=10)],
    )
    fig.write_html(get_output_path("html"))


if __name__ == "__main__":
    main()
