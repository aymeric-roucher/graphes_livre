import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from graphes_livre.utils import SEA_BLUE, apply_template, get_output_path

data_clean = pd.read_json("data/benchmarks_clean.json")

benchmarks_df = data_clean
print(benchmarks_df.head())

benchmarks_df = benchmarks_df.dropna(subset=["name"])
benchmarks_df["publication_date"] = pd.to_datetime(benchmarks_df["publication_date"])
benchmarks_df["baseline_beaten_date"] = pd.to_datetime(
    benchmarks_df["baseline_beaten_date"]
)

current_date = pd.to_datetime("2025-09-01")
benchmarks_df = benchmarks_df.loc[~benchmarks_df["baseline_beaten_date"].isna()]

benchmarks_df = benchmarks_df.sort_values("publication_date")

fig = go.Figure()

colors = px.colors.qualitative.Set3

PLOT_EXPONENTIALS = False

if PLOT_EXPONENTIALS:
    benchmarks_df = benchmarks_df.loc[~benchmarks_df["baseline_beaten_date"].isna()]
    benchmarks_df["duration"] = (
        benchmarks_df["baseline_beaten_date"] - benchmarks_df["publication_date"]
    ).apply(lambda timedelta: timedelta.days)
    benchmarks_df["celerity"] = 1 / benchmarks_df["duration"]
    for i, row in benchmarks_df.iterrows():
        fig.add_trace(
            go.Scatter(
                name=row["name"],
                x=[row["publication_date"], row["publication_date"]],
                y=[0, row["celerity"]],
                mode="lines",
            )
        )
    fig.write_image("celerity.png")

# Sort by publication date to ensure proper chronological order
benchmarks_df_sorted = benchmarks_df.sort_values("publication_date").reset_index(
    drop=True
)

for i, row in benchmarks_df_sorted.iterrows():
    pub_date = row["publication_date"]
    end_date = row["baseline_beaten_date"]
    y_position = len(benchmarks_df_sorted) - i - 1  # Reverse order: earliest on top

    # Main trace for the bar
    fig.add_trace(
        go.Scatter(
            x=[pub_date, end_date],
            y=[y_position + 1, y_position + 1],
            mode="lines",
            line=dict(width=12, color=SEA_BLUE),
            name="",
            showlegend=False,
            hovertemplate=f"<b>{row['name']}</b><br>"
            + f"Publication: {pub_date.strftime('%Y-%m')}<br>"
            + f"Baseline beaten: {end_date.strftime('%Y-%m')}<br>"
            + f"Duration: {(end_date - pub_date).days} days<extra></extra>",
        )
    )

    fig.add_annotation(
        x=end_date,
        y=y_position + 1,
        text=f"{row['name']} - {row['short_french_high_level_name']}",
        showarrow=False,
        font=dict(color="black", size=11, family="Open Sans"),
        xanchor="left",
        xshift=2,
    )

min_date = benchmarks_df_sorted["publication_date"].min()
max_date = current_date
print(min_date, max_date)


apply_template(fig)
range_years = range(1990, 2026, 5)
fig.update_layout(
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        range=[0.5, len(benchmarks_df_sorted) + 0.5],
        linewidth=0,
    ),
    xaxis=dict(
        tickvals=[pd.to_datetime(f"{year}-01-01") for year in range_years],
        ticktext=[str(year) for year in range_years],
        tickfont=dict(size=12),
        range=[min_date - pd.DateOffset(months=3), max_date + pd.DateOffset(months=3)],
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
    ),
    showlegend=False,
    margin=dict(l=5, t=0, b=20, r=235),
)
fig.write_image(get_output_path("jpg"), width=450, height=320, scale=4)


# NOTES - I removed:
# - PIQA for low quality (Reported human baseline is 95% (https://arxiv.org/abs/1911.11641), but scores ceiling at 90% since 2022 is highly dubious)
# - HellaSwag for low quality: https://www.surgehq.ai/blog/hellaswag-or-hellabad-36-of-this-popular-llm-benchmark-contains-errors
# - LibriSpeech because no human baseline
# - SWE-Bench because no human baseline (more info: https://github.com/SWE-bench/SWE-bench/issues/72)
# - HLE because no human baseline
# - SuperGLUE because human baseline was already beaten at launch
# - BigBench because human baseline was already beaten at launch
# {
#     "name": "BigBench",
#     "short_french_high_level_name": "Raisonnement complexe",
#     "publication_date": "2022-06-09",
#     "baseline_beaten_date": "2022-10-17",
#     "source_for_when_the_baseline_was_beaten": "Baseline beaten https://arxiv.org/abs/2210.09261"
# },
# - HumanEval because no human baseline
# Switchboard because too old:
#     {
#         "name": "Switchboard",
#         "short_french_high_level_name": "Reconnaissance vocale",
#         "publication_date": "1992-03-23",
#         "baseline_beaten_date": "2017-03-15",
#         "source_for_when_the_baseline_was_beaten": "https://www.microsoft.com/en-us/research/blog/human-parity-speech-recognition-achieved/"
#     },
