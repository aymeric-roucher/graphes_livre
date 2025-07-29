from datetime import datetime

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
            title_font_size=16,
            # title_font_weight="bold",
            tickfont_size=12,
        ),
    )
    fig.update_layout(margin=dict(l=60, r=40, t=40, b=40))
    fig.write_image(get_output_path("jpg"), width=700, height=550, scale=4)


if __name__ == "__main__":
    main()
