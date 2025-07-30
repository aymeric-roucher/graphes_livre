import numpy as np
import pandas as pd
import plotly.express as px

from graphes_livre import SEA_BLUE, apply_template, get_output_path

# Source: https://www.singularity.com/charts/page67.html

# Define the data
data_original = [
    [1900, 0.000005821, "Machine analytique de Babbage"],
    [1908, 0.0001299, "Hollerith Tabulator"],
    [1911, 0.00005787, "Monroe Calculator"],
    [1919, 0.001064, "IBM Tabulator"],
    [1928, 0.0006993, "National Ellis 3000"],
    [1939, 0.008547, "Zuse 2"],
    [1940, 0.01431, "Bell Calculator Model 1"],
    [1941, 0.0463, "Zuse 3"],
    [1943, 5.308, "Colossus"],
    [1946, 0.7981, "ENIAC"],
    [1948, 0.3698, "IBM SSEC"],
    [1949, 1.837, "BINAC"],
    [1949, 1.044, "EDSAC"],
    [1951, 1.43, "Univac I"],
    [1953, 6.104, "Univac 1103"],
    [1953, 11.88, "IBM 701"],
    [1954, 0.3669, "EDVAC"],
    [1955, 16.45, "Whirlwind"],
    [1955, 3.438, "IBM 704"],
    [1958, 0.3257, "Datamatic 1000"],
    [1958, 0.9144, "Univac II"],
    [1960, 1.514, "IBM 1620"],
    [1960, 151.5, "DEC PDP-1"],
    [1961, 282.5, "DEC PDP-4"],
    [1962, 29.43, "Univac III"],
    [1964, 158.6, "CDC 6600"],
    [1965, 482.6, "IBM 1130"],
    [1965, 1792, "DEC PDP-8"],
    [1966, 49.72, "IBM 360 Model 75"],
    [1968, 213.6, "DEC PDP-10"],
    [1973, 728.6, "Intellec-8"],
    [1973, 3401, "Data General Nova"],
    [1975, 10580, "Altair 8800"],
    [1976, 777, "DEC PDP-11 Model 70"],
    [1977, 3720, "Cray 1"],
    [1977, 26870, "Apple II"],
    [1979, 1114, "DEC VAX 11 Model 780"],
    [1980, 5621, "Sun-1"],
    [1982, 126600, "IBM PC"],
    # [1982, 126600, "Compaq Portable"],
    [1983, 86280, "IBM AT-80286"],
    [1984, 85030, "Apple Macintosh"],
    [1986, 538200, "Compaq Deskpro 386"],
    [1987, 232600, "Apple Mac II"],
    [1993, 3549000, "Pentium PC"],
    [1996, 48080000, "Pentium PC"],
    [1998, 133300000, "Pentium II"],
]
data_self_made = [
    [2002, 0.4e12 / 400, "Radeon 9700 Pro"],
    [2003, 596 * 1e12 / 1.3e6, "IBM Blue Gene/L"],
    [2006, 345 * 1e9 / 600, "GeForce 8800 GX"],
    [2009, 2.75 * 1e12 / 700, "Radeon HD5970"],
    [2010, 1.344 * 1e12 / 500, "GTX 480"],
    [2013, 4.5e12 / 1000, "GTX Titan"],
    [2016, 10.974 * 1e12 / 1200, "NVIDIA Titan X"],
    [2020, 312 * 1e12 / 20000, "NVIDIA A100"],
    [2023, 1979 * 1e12 / 30000, "NVIDIA H100"],
    [2024, 1.8 * 1e16 / 30000, "NVIDIA B100"],
]


data = pd.DataFrame(
    data_original + data_self_made, columns=["Year", "CPS/$1", "Device"]
)
# Create DataFrame
df = data
# Create the scatterplot
fig = px.scatter(
    df,
    x="Year",
    y="CPS/$1",
    hover_name="Device",
    labels={"CPS/$1": "FLOPs pour 1$"},
    log_y=True,
)

selected_devices = [
    "Machine analytique de Babbage",
    "ENIAC",
    "IBM 701",
    "Apple II",
    "IBM PC",
    "Pentium II",
    "GTX Titan",
    "NVIDIA B100",
]
for _, row in df[df["Device"].isin(selected_devices)].iterrows():
    fig.add_annotation(
        x=row["Year"],
        y=np.log10(row["CPS/$1"]),
        text=row["Device"],
        showarrow=False,
        yshift=0 if row["Device"] == "Machine analytique de Babbage" else 15,
        xshift=60 if row["Device"] == "Machine analytique de Babbage" else 0,
        font=dict(size=13),
    )

# Customize the plot
fig.update_traces(marker=dict(size=8, opacity=0.7, color=SEA_BLUE))

apply_template(fig, width=700, height=500)

# Determine the min and max y values for the range
y_min = df["CPS/$1"].min()
y_max = df["CPS/$1"].max()


# Since log_y=True, use logspace for ticks
fig.update_layout(
    yaxis_range=[-6, 12],
)
fig.update_yaxes(dtick=3, tickformat=".0e")
fig.update_xaxes(title="")
# Show the plot
fig.write_image(get_output_path("jpg"), width=700, height=500, scale=4)
