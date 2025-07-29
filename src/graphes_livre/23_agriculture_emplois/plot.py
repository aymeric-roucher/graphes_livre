# Source for data: https://ourworldindata.org/grapher/number-of-people-employed-in-agriculture

import pandas as pd
import plotly.express as px
from graphes_livre import apply_template, get_output_path

BEGIN_YEAR = 1900

df = pd.read_csv("data/number-of-people-employed-in-agriculture.csv")
print(df.dtypes)

selected_countries = ["France", "Japan", "Spain", "United States", "United Kingdom"]
df = df.loc[(df["Entity"].isin(selected_countries)) & (df["Year"] >= 1900)]
df["Entity"] = df["Entity"].map(
    {
        "Japan": "Japon",
        "France": "France",
        "Spain": "Espagne",
        "United Kingdom": "Royaume-Uni",
        "United States": "Etats-Unis",
    }
)

fig = px.line(
    x=df["Year"],
    y=df["number_employed_agri"] / 1e6,
    color=df["Entity"],
)

apply_template(fig)
# Update layout
fig.update_layout(width=700, height=450, showlegend=False, xaxis_title=None)
fig.update_yaxes(
    title="Nombre d'employés (millions)",
    range=[0, 18],
    tickvals=[5, 10, 15],
    ticktext=["5", "10", "15"],
)
# Show the plot
fig.write_image(get_output_path("jpg"), width=700, height=450, scale=4)
