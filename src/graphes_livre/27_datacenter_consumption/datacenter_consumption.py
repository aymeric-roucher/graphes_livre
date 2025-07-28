import pandas as pd
import plotly.express as px

from graphes_livre import SEA_BLUE, apply_template, get_output_path

datacenter_consumption = {
    2000: 13.95147313691507,
    2000: 19.410745233968782,
    2001: 23.656845753899518,
    2002: 26.689774696707104,
    2003: 28.509532062391713,
    2004: 30.329289428076265,
    2005: 33.36221837088385,
    2006: 38.214904679376104,
    2007: 46.100519930675944,
    2008: 55.19930675909882,
    2009: 65.51126516464473,
    2010: 80.06932409012131,
    2011: 96.44714038128251,
    2013: 118.89081455805893,
    2014: 140.12131715771233,
    2015: 156.4991334488735,
    2016: 171.66377816291163,
    2017: 191.07452339688044,
    2017: 211.09185441941077,
    2019: 232.92894280762565,
    2019: 259.01213171577126,
    2020: 289.34142114384747,
    2022: 317.8509532062392,
    2023: 339.0814558058926,
    2024: 351.81975736568455,
}
datacenter_consumption = pd.Series(datacenter_consumption).reset_index()
datacenter_consumption.columns = ["Année", "Consommation (TWh)"]

fig = px.line(datacenter_consumption, x="Année", y="Consommation (TWh)")
fig.update_traces(line_color=SEA_BLUE)
apply_template(fig)

fig.update_layout(width=500)

ITALY_CONSUMPTION = 306
# Add horizontal line for Italy's consumption
fig.add_shape(
    type="line",
    x0=fig.data[0].x[0],  # Start of x-axis
    x1=fig.data[0].x[-1],  # End of x-axis
    y0=ITALY_CONSUMPTION,
    y1=ITALY_CONSUMPTION,
    line=dict(color="gray", width=1),
)

# Add annotation for Italy
fig.add_annotation(
    x=2008,  # Place annotation at the end of x-axis
    y=ITALY_CONSUMPTION + 10,
    text="Consommation de l'Italie",
    # xshift=20,  # Shift label slightly to the right
    showarrow=False,
    yanchor="middle",
    font=dict(color="gray", style="italic"),
)
fig.update_xaxes(
    range=[2000, 2024],
    tick0=2000,
    dtick=4,
)

fig.write_html(get_output_path("html"))
