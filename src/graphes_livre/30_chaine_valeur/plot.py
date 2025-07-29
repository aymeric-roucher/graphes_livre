import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from graphes_livre import apply_template, get_output_path

# Data from the semiconductor value chain table (2021, in percent)
data = {
    "Activity": [
        "Conception logique, fabless",
        "Conception mémoire, intégrée",
        "Conception (Dao), fab-lite",
        "Équipement de fabrication",
        "Matériaux",
        "Fabrication de wafers",
        "Assemblage, emballage & test",
        "Global",
    ],
    "États-Unis": [67, 28, 37, 42, 10, 11, 5, 35],
    "Europe": [8, 0, 18, 21, 6, 9, 4, 10],
    "Chine": [6, 0, 9, 0, 19, 21, 38, 11],
    "Corée du Sud": [4, 58, 6, 3, 17, 17, 9, 16],
    "Japon": [4, 8, 21, 27, 14, 16, 6, 13],
    "Taïwan": [9, 4, 4, 0, 23, 19, 19, 10],
}

df = pd.DataFrame(data)
print(df)
df = df.drop(index=[2])

df_chart = df.copy()

# Define colors using Plotly qualitative palette
regions = ["États-Unis", "Europe", "Chine", "Corée du Sud", "Japon", "Taïwan"]
colors = dict(zip(regions, px.colors.qualitative.Plotly))

# Create the stacked horizontal bar chart
fig = go.Figure()

# Add bars for each region
for region in regions:
    fig.add_trace(
        go.Bar(
            name=region,
            y=df_chart["Activity"],
            x=df_chart[region],
            orientation="h",
            marker_color=colors[region],
            showlegend=True,
        )
    )

fig.update_layout(
    barmode="stack",
    xaxis_title=None,
    yaxis_title=None,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        itemwidth=60,  # Increased spacing between items to prevent cutoff
        tracegroupgap=10,  # Gap between groups
    ),
    margin=dict(l=250, r=20, t=60, b=30),  # Increased top margin for legend
)

# Update x-axis
fig.update_xaxes(
    range=[0, 101],
    tickvals=[0, 20, 40, 60, 80, 100],
    ticktext=["0", "20", "40", "60", "80", "100"],
    showline=False,
    showticklabels=True,
    showgrid=True,
    gridwidth=2,
    gridcolor="lightgray",
)

# Update y-axis to reverse order (to match the original chart)
fig.update_yaxes(
    categoryorder="array", categoryarray=df_chart["Activity"].tolist()[::-1]
)

# Add horizontal line to separate Overall from other categories
fig.add_shape(
    type="line",
    x0=-0.5,  # Start before the plot area
    x1=0.99,  # Extend across plot area
    y0=0.5,  # Position between Overall and first activity
    y1=0.5,
    xref="paper",  # Use paper coordinates to extend beyond plot
    yref="y",  # Use data coordinates for y-axis
    line=dict(color="black", width=1, dash="solid"),
)

apply_template(fig, width=750, height=300)
# Save the plot
fig.write_image(get_output_path("jpg"), width=750, height=300, scale=4)
