# Source for data: https://ourworldindata.org/grapher/number-of-people-employed-in-agriculture

import pandas as pd
import plotly.express as px

from graphes_livre import apply_template, get_output_path

BEGIN_YEAR = 1900

df = pd.read_csv("data/number-of-people-employed-in-agriculture.csv")
print(df.dtypes)

country_names = {
    "Japan": "Japon",
    "France": "France",
    "Spain": "Espagne",
    "United Kingdom": "Royaume-Uni",
    "United States": "Etats-Unis",
}
selected_countries = ["France", "Japan", "Spain", "United States", "United Kingdom"]
df = df.loc[(df["Entity"].isin(selected_countries)) & (df["Year"] >= 1900)]
df["Entity"] = df["Entity"].map(country_names)

fig = px.line(
    x=df["Year"],
    y=df["number_employed_agri"] / 1e6,
    color=df["Entity"],
)

apply_template(fig)
# Update layout
fig.update_layout(
    width=700,
    height=450,
    showlegend=False,
    xaxis_title=None,
    margin=dict(l=60, r=40, t=40, b=40),
)
fig.update_yaxes(
    title="Nombre d'employés (millions)",
    range=[0, 18],
    tickvals=[5, 10, 15],
    ticktext=["5", "10", "15"],
)
# Create mapping from country name to trace index
country_to_trace = {trace.name: i for i, trace in enumerate(fig.data)}

for country_name in country_names.values():
    height = (
        df[(df["Entity"] == country_name) & (df["Year"].isin(list(range(1900, 1911))))][
            "number_employed_agri"
        ].mean()
        / 1e6
    )
    # Get the color from the corresponding trace
    trace_index = country_to_trace[country_name]
    line_color = fig.data[trace_index].line.color
    fig.add_annotation(
        x=1902,
        y=height,
        text=country_name,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=13, color=line_color, weight="bold"),
        yshift=10,
    )
# Show the plot
fig.write_image(get_output_path("jpg"), width=700, height=450, scale=4)
