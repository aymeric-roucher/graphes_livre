from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from graphes_livre import SEA_BLUE, apply_template, get_output_path


def main():
    df = pd.read_csv("data/notable_ai_models.csv")

    df = df.loc[df["Publication date"] > "2018-01-01"]

    df = df.sort_values("Publication date")
    df["Max compute until now"] = df["Training compute (FLOP)"].cummax()
    df = df.loc[df["Training compute (FLOP)"] >= df["Max compute until now"] / 100]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Publication date"],
            y=df["Training compute (FLOP)"],
            text=df["Model"],
            mode="markers",
            marker=dict(color=SEA_BLUE),
        )
    )

    # Add trend line: 0.6 ODG/year (ODG = Orders of magnitude)
    # Line goes from 2019 to 2025
    trend_years = np.array([2018, 2026])
    # Starting point around 10^21, growing by 0.6 orders of magnitude per year
    base_log = 22.3  # in 2019
    trend_y = 10 ** (base_log + 0.64 * (trend_years - 2019))

    fig.add_trace(
        go.Scatter(
            x=trend_years,
            y=trend_y,
            mode="lines",
            line=dict(dash="dash", color="black", width=2),
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Add specific model annotations
    annotations = [
        {
            "model": "GPT-3",
            "date": "2020-05-28",
            "compute": 23.314,
        },
        {
            "model": "GPT-4",
            "date": "2023-03-15",
            "compute": 25.12,
        },
        {
            "model": "Gemini 1.0 Ultra",
            "date": "2023-12-06",
            "compute": 25.5,
        },
        {
            "model": "Grok 4",
            "date": "2025-07-09",
            "compute": 26.5,
        },
    ]

    for ann in annotations:
        fig.add_annotation(
            x=ann["date"],
            y=ann["compute"],
            text=ann["model"],
            showarrow=False,
            xshift=0,
            yshift=15,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13, color="black"),
        )

    # Add trend line annotation
    fig.add_annotation(
        x=datetime(2021, 12, 1),
        y=24.35,
        text="Tendance : ~0.6 ODG/an",
        showarrow=False,
        textangle=-26,
        font=dict(size=16, color="black", weight="bold", style="italic"),
    )

    fig.update_layout(
        xaxis=dict(
            range=[datetime(2018, 7, 15), datetime(2025, 7, 30)],
            tickvals=list(range(2018, 2026)),
            tickformat="%Y",
        ),
        yaxis=dict(
            type="log",
            range=[20, 27],
            dtick=1,
            # tickvals=[1e18, 1e20, 1e22, 1e24],
            # tickformat=",.0f",
        ),
    )

    apply_template(fig, width=800, height=600)
    fig.update_layout(
        yaxis=dict(
            title="Puissance d'entraînement (FLOP)",
            # title_font_weight="bold",
        ),
        showlegend=False,
    )
    fig.update_layout(margin=dict(l=60, r=40, t=40, b=40))
    fig.update_xaxes(tickfont_size=14)
    fig.update_yaxes(tickfont_size=14)
    fig.write_html("epoch_progress.html")
    fig.write_image(get_output_path("jpg"), width=650, height=500, scale=4)


if __name__ == "__main__":
    main()
