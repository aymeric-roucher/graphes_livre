import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from graphes_livre import SEA_BLUE, apply_template, get_output_path


def create_s_curve():
    t = np.linspace(0, 10, 200)

    # Create S-curve: exponential then linear
    transition_point = 3
    exp_part = t <= transition_point
    linear_part = t > transition_point

    progress = np.zeros_like(t)

    # Exponential part
    progress[exp_part] = 0.1 * np.exp(t[exp_part])

    # Linear part (continuous at transition)
    exp_value = 0.1 * np.exp(transition_point)
    progress[linear_part] = exp_value + 0.1 * np.exp(transition_point) * (
        t[linear_part] - transition_point
    )

    return t, progress


def create_gaussian(t, center=6.87, amplitude=2, width=0.35):
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


# Generate data
t, progress = create_s_curve()
gaussian_curve = create_gaussian(t)

# Create subplots with different heights
fig = make_subplots(rows=1, cols=1, vertical_spacing=0.05, row_heights=[1])

# First plot: S-curve with horizontal dashed lines
split_point = 6.4
split_idx = np.argmin(np.abs(t - split_point))

# Continuous part (before split)
fig.add_trace(
    go.Scatter(
        x=t[: split_idx + 1],
        y=progress[: split_idx + 1],
        mode="lines",
        name="Intelligence",
        line=dict(width=3, color=SEA_BLUE),
    ),
    row=1,
    col=1,
)

# Dashed part (after split)
fig.add_trace(
    go.Scatter(
        x=t[split_idx:],
        y=progress[split_idx:],
        mode="lines",
        name="Intelligence",
        line=dict(width=3, color=SEA_BLUE, dash="dash"),
    ),
    row=1,
    col=1,
)

# Black point at split
fig.add_trace(
    go.Scatter(
        x=[t[split_idx]],
        y=[progress[split_idx]],
        mode="markers",
        name="Split point",
        marker=dict(size=8, color="black"),
    ),
    row=1,
    col=1,
)

# Add horizontal dashed lines at different y values with labels
labels = ["Corbeau", "Chimpanzé", "Enfant", "Bachelier", "Prix Nobel"]
y_vals = [2, 6, 8.5, 9.2, 10.5]

# Add shaded area between "Enfant" and "Prix Nobel"
fig.add_shape(
    type="rect",
    x0=0,
    x1=8.7,
    y0=y_vals[2],
    y1=y_vals[4],  # Between "Enfant" and "Prix Nobel"
    fillcolor="lightblue",
    opacity=0.2,
    line=dict(width=0),
    row=1,
    col=1,
)

for i, y_val in enumerate(y_vals):
    color = "gray" if i < 2 else "black"
    fig.add_hline(y=y_val, line_color=color, opacity=0.7, row=1, col=1, line_width=1)
    fig.add_annotation(
        x=1,
        y=y_val + 0.3,
        text=labels[i],
        showarrow=False,
        font=dict(size=17, color=color, style="italic"),
        xref="x",
        yref="y",
        row=1,
        col=1,
        xanchor="left",
    )

# Second plot: Angular speed with split at same point
# Continuous part (before split)
# fig.add_trace(
#     go.Scatter(
#         x=t[:split_idx+1],
#         y=gaussian_curve[:split_idx+1],
#                         mode='lines',
#                         name='Vitesse perçue',
#         line=dict(width=3, color='red')
#     ),
#     row=2, col=1
# )

# Dashed part (after split)
# fig.add_trace(
#     go.Scatter(
#         x=t[split_idx:],
#         y=gaussian_curve[split_idx:],
#         mode='lines',
#         name='Vitesse perçue',
#         line=dict(width=3, color='red', dash='dash')
#     ),
#     row=2, col=1
# )

# # Black point at split
# fig.add_trace(
#     go.Scatter(
#         x=[t[split_idx]],
#         y=[gaussian_curve[split_idx]],
#         mode='markers',
#         name='Split point',
#         marker=dict(size=8, color='black')
#     ),
#     row=2,
#     col=1
# )

# Add arrow at top of Y axis for Intelligence
fig.add_annotation(
    x=0,
    y=13,
    ax=0,
    ay=-0.13,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    showarrow=True,
    row=1,
    col=1,
)

# Add arrow at right of X axis for Time
fig.add_annotation(
    x=8.7,
    y=0,
    ax=-0.1,
    ay=0,
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="black",
    showarrow=True,
    row=1,
    col=1,
)

# Add arrows for bottom subplot
# Y axis arrow
# fig.add_annotation(
#     x=0, y=2.3,
#     ax=0, ay=-0.13,
#     xref="x2", yref="y2",
#     axref="x2", ayref="y2",
#     arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black",
#     showarrow=True,
#     row=2, col=1
# )

# # X axis arrow
# fig.add_annotation(
#     x=8.7, y=0,
#     ax=-0.1, ay=0,
#     xref="x2", yref="y2",
#     axref="x2", ayref="y2",
#     arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black",
#     showarrow=True,
#     row=2, col=1
# )

# Update layout
fig.update_layout(showlegend=False)

# Remove tick labels, grid lines and tick marks for both plots
fig.update_xaxes(
    showticklabels=False,
    showgrid=False,
    title_text="Temps",
    title_standoff=5,
    row=1,
    col=1,
)
# fig.update_xaxes(showticklabels=False, showgrid=False,title_text="Temps", title_standoff=5, row=2, col=1)
fig.update_yaxes(
    showticklabels=False,
    showgrid=False,
    title_text="Intelligence",
    title_standoff=5,
    row=1,
    col=1,
    range=[0, 13],
)
# fig.update_yaxes(showticklabels=False, showgrid=False,title_text="Vitesse perçue", title_standoff=5, row=2, col=1, range=[0, 2.3])
fig.update_xaxes(range=[0, 8.7])
apply_template(fig, width=600, height=600)

fig.write_html(get_output_path("html"))
