import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from graphes_livre.utils import apply_template, get_output_path

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


# Dynamic slot assignment - slots become available after benchmark is beaten
import random

NUM_SLOTS = 7
available_slots = list(range(NUM_SLOTS))
slot_assignments = {}
slot_end_dates = {}


names_to_show_in_legend = [
    "GPQA-diamond",
    "GSM8K",
    "GLUE",
    "SQuAD1.1",
    "SQuAD2.0",
    "CommonsenseQA",
    "MMMU",
    "MedQA",
]

# Special characters for each benchmark in legend
special_chars = ["★", "●", "▲", "■", "◆", "✦", "⬢", "᛭"]
legend_char_map = {
    name: special_chars[i] for i, name in enumerate(names_to_show_in_legend)
}
# Sort by publication date to process chronologically
for i, row in benchmarks_df.iterrows():
    pub_date = row["publication_date"]
    end_date = row["baseline_beaten_date"]

    # Free up slots that have ended before this publication date
    slots_to_free = []
    for slot, slot_end in slot_end_dates.items():
        if slot_end <= pub_date:
            slots_to_free.append(slot)

    for slot in slots_to_free:
        if slot not in available_slots:
            available_slots.append(slot)
        del slot_end_dates[slot]

    # Assign a random available slot
    if available_slots:
        if row["name"] == "Switchboard":
            y_position = NUM_SLOTS - 1
        else:
            y_position = random.choice(available_slots)
        available_slots.remove(y_position)
    else:
        raise ValueError("No slots available, increase parameter NUM_SLOTS")

    slot_assignments[row["name"]] = y_position
    slot_end_dates[y_position] = end_date

    # Create the main trace for the bar
    fig.add_trace(
        go.Scatter(
            x=[pub_date, end_date],
            y=[y_position + 1, y_position + 1],
            mode="lines",
            line=dict(width=12, color=colors[i % len(colors)]),
            name="",
            showlegend=False,
            hovertemplate=f"<b>{row['name']}</b><br>"
            + f"Publication: {pub_date.strftime('%Y-%m')}<br>"
            + f"Baseline beaten: {end_date.strftime('%Y-%m')}<br>"
            + f"Duration: {(end_date - pub_date).days} days<extra></extra>",
        )
    )

    # Add a separate invisible trace for legend (text only)
    if row["name"] in names_to_show_in_legend:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=0, opacity=0),
                name=f"{legend_char_map[row['name']]} {row['name']}",
                showlegend=True,
            )
        )

    # Add text annotation in the middle of the bar
    mid_date = pub_date + (end_date - pub_date) / 2
    if row["name"] in names_to_show_in_legend:
        # Show special character for legend items
        annotation_text = legend_char_map[row["name"]]
    else:
        # Show full name for non-legend items
        annotation_text = row["name"]

    fig.add_annotation(
        x=mid_date,
        y=y_position + 1,
        text=annotation_text,
        showarrow=False,
        font=dict(color="black", size=10, family="Open Sans"),
    )

min_date = benchmarks_df["publication_date"].min()
max_date = current_date
print(min_date, max_date)

# Determine the maximum slot used for layout
max_slot_used = max(slot_assignments.values()) if slot_assignments else 0


apply_template(fig)
range_years = range(1990, 2026, 5)
fig.update_layout(
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        range=[0.5, max_slot_used + 1.5],
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
    showlegend=True,
    legend=dict(
        font=dict(size=10),
        x=0.9,
        y=1,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.)",
    ),
)
fig.write_image(get_output_path("jpg"), width=600, height=300, scale=4)


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
