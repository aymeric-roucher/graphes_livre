from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

from graphes_livre import apply_template, get_output_path


# Sigmoid function
def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-(a * x + b)))


def compute_scaling_law():
    # Define your two points (x1, y1) and (x2, y2)
    x2, y2 = 3.8 * 1e25, 0.40  # Point 2 for Llama-3.1-405B
    x1, y1 = (
        x2 * 7 / 30.84,
        0.25,
    )  # Point 1 from Llama-3.1 70B, on this page for the rule of 3: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct
    x0, y0 = (
        x2 * 1.46 / 30.84,
        0.0,
    )  # Point 0 from Llama-3.1 8B, on this page for the rule of 3: https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct

    x1_log, x2_log = np.log(x1), np.log(x2)

    a = (-np.log(1 / y1 - 1) + np.log(1 / y2 - 1)) / (x1_log - x2_log)
    b = -np.log(1 / y2 - 1) - a * x2_log

    # Generate x values for the plot (log scale)
    x_values = np.exp(np.linspace(np.log(x1) - 10, np.log(x2) + 10, 500, endpoint=True))

    # Calculate the corresponding y values using the fitted sigmoid function
    y_values = np.array([sigmoid(np.log(x), a, b) for x in x_values])

    # Plot using Plotly
    fig = go.Figure()

    # Add the sigmoid curve
    fig.add_trace(
        go.Scatter(x=x_values, y=y_values, mode="lines", name="Courbe en S extrapolée")
    )

    # Add the original points
    fig.add_trace(
        go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode="markers",
            name="Points de référence",
            marker=dict(size=15),
        )
    )

    # Update the layout to set log scale for x-axis
    fig.update_layout(
        # title="Lois d'échelle pour la performance sur GAIA",
        xaxis=dict(type="log", title="Training compute"),
        yaxis=dict(title="Score", range=[0, 1], tickformat=",.0%"),
        showlegend=True,
    )
    apply_template(fig, width=800, height=600)

    # Show the plot
    fig.write_image(get_output_path("jpg"), width=800, height=600, scale=4)


def plot_gaia_time_law():
    data = pd.DataFrame(
        [
            # ["GPT-4-Turbo", "2023-11-14", 6.7],
            # ["GPT-4-Turbo", "2024-02-22", 12.96],
            ["GPT-4-Turbo - FRIDAY", "2024-01-24", 24],
            ["GPT-4o - smolagents", "2024-06-27", 33],
            ["GPT-4o - Dynasaur", "2024-10-04", 38.21],
            ["Claude-3.5 - Langfun", "2024-12-02", 49.3],
            ["Claude-3.5 - h2oGPTe", "2025-01-29", 65],
            ["Claude-3.7 - h2oGPTe", "2025-03-20", 74],
            ["o3 - Shu Zero", "2025-06-26", 80.4],
        ],
        columns=["model", "date", "score"],
    )

    # Convert dates to datetime
    data["date"] = pd.to_datetime(data["date"])

    # Convert dates to numerical values for fitting
    base_date = datetime(2022, 1, 1)
    x_data = np.array([(date - base_date).days for date in data["date"]])
    y_data = np.array(data["score"].values)

    SEA_BLUE = "rgba(0,100,255,0.5)"

    # Proper 0-100 sigmoid function with just midpoint and width
    def sigmoid_time(x, x0, k):
        return 100 / (1 + np.exp(-k * (x - x0)))

    # Fit sigmoid curve with just two parameters
    p0 = [np.mean(x_data), 0.002]  # Initial parameters guess: [midpoint, steepness]
    popt, _ = curve_fit(sigmoid_time, x_data, y_data, p0=p0, maxfev=5000)

    # Generate smooth curve going far enough to see it reach ~95
    x_smooth = np.linspace(min(x_data) - 500, max(x_data) + 1200, 1000)
    y_smooth = sigmoid_time(x_smooth, *popt)

    # Convert back to dates for plotting
    dates_smooth = [base_date + timedelta(days=int(x)) for x in x_smooth]

    # Create the plot
    fig = go.Figure()

    # Add the sigmoid curve
    fig.add_trace(
        go.Scatter(
            x=dates_smooth,
            y=y_smooth,
            mode="lines",
            name="Courbe sigmoide ajustée",
            line=dict(color=SEA_BLUE),
            showlegend=False,
        )
    )

    # Add the data points RIGHT AFTER the curve
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["score"],
            mode="markers",
            name="Performance du meilleur agent",
            marker=dict(
                size=10,
                color=SEA_BLUE,
                symbol="circle",
            ),
            showlegend=False,
        )
    )

    # Add a horizontal dashed black line at y=92%
    fig.add_trace(
        go.Scatter(
            x=[dates_smooth[0], dates_smooth[-1]],
            y=[92, 92],
            mode="lines",
            name="Ligne 92%",
            line=dict(color="black", dash="dash"),
            showlegend=False,
        )
    )

    # Add the line text
    fig.add_trace(
        go.Scatter(
            x=[2025],
            y=[93],
            mode="text",
            marker=dict(size=10, color="black", symbol="square"),
            text=["Performance humaine: 92%"],
            textposition="top center",
            textfont=dict(size=13),
            showlegend=False,
        )
    )
    print(data)
    print("Date range in data:", data["date"].min(), "to", data["date"].max())
    print("Score range in data:", data["score"].min(), "to", data["score"].max())

    x_years = [2024, 2025, 2026]
    x_ticks = [datetime(year, 1, 1) for year in x_years]

    fig.update_layout(
        # title="Performance des assistants généralistes au fil du temps",
        # xaxis_title="Année",
        yaxis_title=dict(text="Score", font_weight="bold"),
        yaxis=dict(range=[0, 101], tickformat=",d"),
        xaxis=dict(
            range=[datetime(2024, 1, 1), datetime(2026, 10, 31)],
            tickvals=x_ticks,
            tickformat="%Y",
        ),
        showlegend=True,
        template="plotly_white",
    )

    # Add grid
    fig.update_xaxes(
        showgrid=True, gridwidth=2, showticklabels=True, ticks="outside", tickwidth=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="LightGrey",
    )
    apply_template(fig, width=600, height=500)
    # fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html(get_output_path("html"))


if __name__ == "__main__":
    plot_gaia_time_law()
