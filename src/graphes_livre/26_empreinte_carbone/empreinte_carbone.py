# the embodied energy of the server per query is 0.03 g CO2e
import pandas as pd
import plotly.express as px

from graphes_livre import SEA_BLUE, apply_template, get_output_path

duration_hours = 0.8
chatgpt_query_g = 3
laptop_footprint_duration_g = 27
desktop_footprint_duration_g = 72

title_emissions = "Emissions du travail d'un jour (g CO2e)"
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
    {"Poste": "Transport en<br>voiture", title_emissions: 8000},
]
footprints = pd.DataFrame.from_dict(footprints, orient="columns")
fig = px.bar(footprints, x="Poste", y=title_emissions, barmode="group")

fig.update_traces(
    texttemplate="%{y:.1f}",  # Format with one decimal place
    textposition="outside",  # Position text outside the bar
    textfont=dict(size=12),  # Set text size
)

apply_template(fig, annotation_text="")
fig.update_layout(width=600, height=500)
fig.update_xaxes(title=None)
fig.update_traces(marker_color=SEA_BLUE)
fig.write_html(get_output_path("html"))
