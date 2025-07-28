import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graphes_livre import apply_template, get_output_path
import numpy as np

FONT_SIZE = 16
data = {
    "variable": [
        # Puissance électrique
        "Puissance électrique<br>(GW)",
        "Puissance électrique<br>(GW)",
        "Puissance électrique<br>(GW)",
        # Financement
        "Financement<br>(milliards USD)",
        "Financement<br>(milliards USD)",
        "Financement<br>(milliards USD)",
        # FLOPs
        "Puissance de<br>calcul (FLOP)",
        "Puissance de<br>calcul (FLOP)",
        "Puissance de<br>calcul (FLOP)",
    ],
    "etat": [
        # Puissance électrique
        "GPT-4 (2022)",
        "Centrale nucléaire",
        "Objectif",
        # Financement
        "GPT-4 (2022)",
        "Stargate (2025)",
        "Objectif",
        # FLOPs
        "GPT-4 (2022)",
        "Production 2024",
        "Objectif",
    ],
    "value": [
        # Puissance électrique
        0.01,
        5,
        10,
        # Financement
        0.1,
        500,
        500,
        # FLOPs
        4 * 1e25,
        1.5
        * 1e6  # NVIDIA produces 1.5M H100 in 2024, each is (power of llama 3 divided by 30.84M H100 hours FLOPS/hour), running for one year
        * 24
        * 365.15
        * 5
        * 1e25
        / (30.84 * 1e6),
        1e28,
    ],
    "category": [
        # Add a category column to maintain order
        "Base",
        "Current",
        "Target",
        "Base",
        "Current",
        "Target",
        "Base",
        "Current",
        "Target",
    ],
}

limites = pd.DataFrame(data)

# Create figure with subplots
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=limites["variable"].unique(),
    vertical_spacing=0.1,  # Add more vertical space for titles
    horizontal_spacing=0.1,
)
fig.update_annotations(
    font_size=FONT_SIZE + 2,
    y=1.05,  # Increase font size  # Move titles up
)

# Color sequence with different opacities
colors = ["rgba(147, 112, 219, " + str(alpha) + ")" for alpha in [1.0, 1.0, 0.5, 0.25]]

# Add bars for each variable
for idx, variable in enumerate(limites["variable"].unique()):
    limites_var = limites.loc[limites["variable"] == variable]
    fig.add_trace(
        go.Bar(
            name=variable,
            x=limites_var["etat"],
            y=limites_var["value"],
            marker_color=colors,
            showlegend=False,
        ),
        row=1,
        col=idx + 1,
    )


# Update all y-axes to log scale and set ranges
yaxis_ranges = {
    "Puissance électrique<br>(GW)": [0.001, 200],
    "Financement<br>(USD)": [0.01, 500],
    "Puissance de<br>calcul (FLOP)": [1e25, 1e28],
}

# Update x-axes
fig.update_xaxes(tickangle=45)

# Update layout
fig.update_layout(
    width=600,
    height=500,
    showlegend=False,
    legend_title_text="État",
    legend={"orientation": "h", "yanchor": "bottom", "y": -0.3},
    barmode="stack",
)
# Add legend for states
fig.add_trace(
    go.Bar(
        name="GPT-4 (2022)", x=[None], y=[None], marker_color=colors[0], showlegend=True
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="Actuel (2024)",
        x=[None],
        y=[None],
        marker_color=colors[1],
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="Difficile à atteindre",
        x=[None],
        y=[None],
        marker_color=colors[2],
        showlegend=True,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        name="Objectif 2030",
        x=[None],
        y=[None],
        marker_color=colors[3],
        showlegend=True,
    ),
    row=1,
    col=1,
)

# Update bar gap and width
fig.update_layout(
    bargap=0.15,
    bargroupgap=0.5,
    margin=dict(
        t=100,  # top margin
        b=150,  # bottom margin (for rotated x-labels)
        l=50,  # left margin
        r=50,  # right margin
    ),
)
apply_template(fig, font_size=FONT_SIZE, width=700)
for idx, (title, ranges) in enumerate(yaxis_ranges.items()):
    fig.update_yaxes(
        type="log",
        range=list(map(np.log10, ranges)),
        dtick=1,
        row=1,
        col=idx + 1,
        tickfont=dict(size=FONT_SIZE - 1),
    )

fig.write_image(get_output_path("png"), scale=3)
