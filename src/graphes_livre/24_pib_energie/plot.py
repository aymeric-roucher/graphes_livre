import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from graphes_livre import SEA_BLUE, apply_template, get_output_path

BEGIN_YEAR = 1965

gdp = pd.read_csv("data/gdp.csv", skiprows=3).drop(
    ["Country Code", "Indicator Code", "Indicator Name"], axis=1
)
gdp = (
    gdp.loc[gdp["Country Name"] == "World"]
    .T.drop(["Country Name", "Unnamed: 68"])[259]
    .rename("gdp")
    .astype(float)
)
gdp.index = gdp.index.astype(int)
gdp = gdp.loc[BEGIN_YEAR:]

energy = pd.read_csv("data/energy.csv").set_index("Year")

df = pd.concat([energy, gdp], axis=1)
df.index.name = "year"

co2 = pd.read_csv("data/co2-emissions-and-gdp.csv")
print(co2.columns)
co2 = co2.loc[
    co2["Entity"] == "World",
    ["Year", "GDP, PPP (constant 2017 international $)", "Annual CO₂ emissions"],
].set_index("Year")

df["co2"] = co2["Annual CO₂ emissions"]

df = df.reset_index()
df["year"] = df["year"].astype(int)
df = df.loc[df["year"] >= 1980]

PLOT_ENGLISH = False

if PLOT_ENGLISH:
    # Assuming df is your existing dataframe
    # Make sure df is sorted by year
    df = df.sort_values("year")

    x_column = "gdp"
    y_column = "co2"
    # Create the scatter plot

    fig = go.Figure(
        go.Scatter(
            x=df[x_column] / 1e12,  # Convert to trillions (billions in French)
            y=df[y_column] / 1e9,  # Convert to billions of tonnes
            mode="lines+markers",  # This tells Plotly to draw both lines and markers
            line=dict(color="black", width=2),
            marker=dict(
                color="black",
                size=5,
                symbol="circle",
                line=dict(color="black", width=2),
            ),
            showlegend=False,
        )
    )

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df[x_column] / 1e12, df[y_column] / 1e9
    )
    line_x = np.array([df[x_column].min() / 1e12, df[x_column].max() / 1e12])
    line_y = slope * line_x + intercept
    print(line_x, line_y)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            name="Ligne X = Y",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            mode="lines",
        )
    )

    apply_template(fig)
    # years_to_annotate = list(range(1990, 2021, 10)) + [2023]
    # for year in years_to_annotate:
    #     year_data = df[df["year"] == year]
    #     if not year_data.empty:
    #         fig.add_annotation(
    #             x=year_data[x_column].iloc[0],
    #             y=year_data[y_column].iloc[0],
    #             text=f"{year:.0f}",
    #             showarrow=False,
    #             yshift=10,
    #             font=dict(size=12),
    #         )

    # Update layout
    fig.update_layout(width=650, height=600)
    fig.update_xaxes(title="GDP (B US$)")
    fig.update_yaxes(title="Emissions de CO2e (Mt)")
    # Show the plot
    fig.write_image(get_output_path("jpg"), width=650, height=600, scale=4)


gdp_name = "GDP, PPP (constant 2017 international $)"
co2_name = "Annual CO₂ emissions"

# Assuming co2 is your existing dataframe
# Make sure df is sorted by year
co2 = co2.reset_index().set_index("Year").sort_index()

# Create the scatter plot

fig = go.Figure(
    go.Scatter(
        x=co2[gdp_name] / 1e12,  # Convert to trillions (billions in French)
        y=co2[co2_name] / 1e9,  # Convert to billions of tonnes
        mode="lines+markers",  # This tells Plotly to draw both lines and markers
        line=dict(color=SEA_BLUE, width=2),
        marker=dict(color=SEA_BLUE, size=5, symbol="circle"),
        showlegend=False,
    )
)


# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    co2[gdp_name] / 1e12, co2[co2_name] / 1e9
)
line_x = np.array([co2[gdp_name].min() / 1e12, co2[gdp_name].max() / 1e12])
line_y = slope * line_x + intercept

# fig.add_trace(
#     go.Scatter(
#         x=line_x,
#         y=line_y,
#         name="Ligne X = Y",
#         line=dict(color="gray", dash="dash"),
#         showlegend=False,
#         mode="lines",
#     )
# )

# Update layout
apply_template(fig)
years_to_annotate = list(range(1990, 2021, 10))
for year in years_to_annotate:
    year_data = co2.loc[year]
    fig.add_annotation(
        x=year_data[gdp_name] / 1e12,
        y=year_data[co2_name] / 1e9,
        text=f"{year:.0f}",
        xshift=0,
        yshift=0,
        showarrow=False,
        font=dict(size=15, color=SEA_BLUE),
        xanchor="left",
        yanchor="top",
    )
fig.update_layout(width=600, height=500, margin=dict(l=50, r=40, t=40, b=50))
fig.update_xaxes(title="PIB mondial (billions de dollars)", title_font=dict(size=17))
fig.update_yaxes(
    title="Emissions mondiales de CO<sub>2</sub>e (Gt)", title_font=dict(size=17)
)

# Show the plot
fig.write_image(get_output_path("jpg"), width=500, height=400, scale=4)
