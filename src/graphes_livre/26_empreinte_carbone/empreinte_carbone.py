# the embodied energy of the server per query is 0.03 g CO2e
import pandas as pd
import plotly.graph_objects as go

from graphes_livre import SEA_BLUE, apply_template, get_output_path

duration_hours = 0.8
chatgpt_query_g = 3
laptop_footprint_duration_g = 27
desktop_footprint_duration_g = 72

title_emissions = "Empreinte quotidienne (gCO2e)"
transport_real_value = 8000
transport_display_value = 1150  # Truncated value for display

footprints = [
    {"Poste": "LLM", title_emissions: chatgpt_query_g / duration_hours * 8},
    {
        "Poste": "Ordinateur<br>portable",
        title_emissions: laptop_footprint_duration_g / duration_hours * 8,
    },
    {
        "Poste": "Ordinateur<br>fixe",
        title_emissions: desktop_footprint_duration_g / duration_hours * 8,
    },
    {"Poste": "Transport en<br>voiture", title_emissions: transport_display_value},
]
footprints = pd.DataFrame.from_dict(footprints, orient="columns")

# Create the base bar chart
fig = go.Figure()

# Add regular bars
for i, row in footprints.iterrows():
    if row["Poste"] == "Transport en<br>voiture":
        # Add the truncated bar with special handling
        fig.add_trace(
            go.Bar(
                x=[row["Poste"]],
                y=[row[title_emissions]],
                marker_color=SEA_BLUE,
                showlegend=False,
                text=[f"{transport_real_value:,.0f}".replace(",", ".")],
                textposition="outside",
            )
        )
    else:
        # Add normal bars
        fig.add_trace(
            go.Bar(
                x=[row["Poste"]],
                y=[row[title_emissions]],
                marker_color=SEA_BLUE,
                showlegend=False,
                text=[f"{row[title_emissions]:,.0f}".replace(",", ".")],
                textposition="outside",
            )
        )

# Add single diagonal cut line to show truncation on the transport bar
transport_bar_x = 3  # Position of transport bar (0-indexed)
cut_height = transport_display_value - 200

# Add single diagonal line to indicate truncation
fig.add_shape(
    type="line",
    x0=transport_bar_x - 0.3,
    x1=transport_bar_x + 0.3,
    y0=cut_height - 30,
    y1=cut_height + 30,
    line=dict(color="white", width=25),
)

fig.update_traces(
    textfont=dict(size=12),
)

apply_template(fig, annotation_text="")
fig.update_layout(
    width=550,
    height=450,
    yaxis_title=title_emissions,
    bargap=0.5,  # Makes bars thinner by increasing gap between them
)
fig.update_xaxes(title=None)
fig.update_yaxes(
    tickvals=[
        0,
        200,
        400,
        600,
        800,
    ],  # Custom tick positions, excluding the highest tick
    range=[0, 1300],
)
fig.write_html(get_output_path("html"))
