import numpy as np
import plotly.graph_objects as go

from graphes_livre import get_output_path


def gaussian_surface(X, Y, center_x, center_y, amplitude, sigma_x, sigma_y):
    return amplitude * np.exp(
        -(
            (X - center_x) ** 2 / (2 * sigma_x**2)
            + (Y - center_y) ** 2 / (2 * sigma_y**2)
        )
    )


x = np.linspace(-3, 4, 1000)
y = np.linspace(0, 6, 1000)
X, Y = np.meshgrid(x, y)

fig = go.Figure()


competences = {
    "Calcul": {
        "x": -2,
        "y": 5,
    },
    "Logique": {
        "x": -0.5,
        "y": 4.5,
    },
    "Traduction": {
        "x": 2,
        "y": 4.75,
    },
    "Recherche web": {
        "x": 0,
        "y": 2,
    },
    "Résumé": {
        "x": 1.5,
        "y": 2.5,
    },
    "Mémoire": {
        "x": 3,
        "y": 2.5,
    },
    "Empathie": {
        "x": 3.5,
        "y": 1,
    },
}

for comp_name, comp in competences.items():
    fig.add_trace(
        go.Scatter3d(
            x=[comp["x"], comp["x"]],
            y=[comp["y"], comp["y"]],
            z=[0.5, 3],
            mode="lines",
            line=dict(color="black", width=3),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[comp["x"]],
            y=[comp["y"]],
            z=[3 - 0.2],
            mode="text",
            text=[comp_name],
            textfont=dict(size=12, color="black"),
            showlegend=False,
        )
    )


Z_blue = (
    0.3 * np.ones_like(X)
    + gaussian_surface(
        X,
        Y,
        center_x=competences["Calcul"]["x"],
        center_y=competences["Calcul"]["y"],
        amplitude=0.3,
        sigma_x=2.0,
        sigma_y=4.0,
    )
    + gaussian_surface(
        X,
        Y,
        center_x=competences["Empathie"]["x"],
        center_y=competences["Empathie"]["y"],
        amplitude=0.3,
        sigma_x=2.0,
        sigma_y=4.0,
    )
)

Z_red = gaussian_surface(
    X,
    Y,
    center_x=competences["Calcul"]["x"],
    center_y=competences["Calcul"]["y"],
    amplitude=2.7,
    sigma_x=0.1,
    sigma_y=0.1,
)

Z_green = (
    gaussian_surface(
        X,
        Y,
        center_x=competences["Logique"]["x"] + 1,
        center_y=competences["Logique"]["y"],
        amplitude=0.7,
        sigma_x=0.7,
        sigma_y=0.5,
    )
    + gaussian_surface(
        X,
        Y,
        center_x=competences["Traduction"]["x"],
        center_y=competences["Traduction"]["y"],
        amplitude=1.0,
        sigma_x=0.7,
        sigma_y=0.4,
    )
    + gaussian_surface(
        X,
        Y,
        center_x=competences["Recherche web"]["x"],
        center_y=competences["Recherche web"]["y"],
        amplitude=0.8,
        sigma_x=0.7,
        sigma_y=0.7,
    )
    + gaussian_surface(
        X,
        Y,
        center_x=competences["Résumé"]["x"],
        center_y=competences["Résumé"]["y"],
        amplitude=1.0,
        sigma_x=0.7,
        sigma_y=0.7,
    )
    + gaussian_surface(
        X,
        Y,
        center_x=1.6,
        center_y=3.9,
        amplitude=0.8,
        sigma_x=0.7,
        sigma_y=0.7,
    )
)

# Set red and green surfaces to NaN where they are below blue surface - 1
Z_red = np.where(Z_red < Z_blue - 0.1, np.nan, Z_red)
Z_green = np.where(Z_green < Z_blue - 0.1, np.nan, Z_green)

fig.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=Z_blue,
        colorscale=[[0, "lightblue"], [1, "blue"]],
        showscale=False,
        name="Base",
    )
)

fig.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=Z_red,
        colorscale=[[0, "red"], [1, "darkred"]],
        showscale=False,
        name="Calcul",
    )
)

fig.add_trace(
    go.Surface(
        x=X,
        y=Y,
        z=Z_green,
        colorscale=[[0, "lightgreen"], [1, "darkgreen"]],
        showscale=False,
        name="Compétences",
    )
)

fig.update_layout(
    title="Compétences en IA",
    scene=dict(
        xaxis_title="",
        yaxis_title="",
        zaxis_title="Performance",
        xaxis=dict(showticklabels=False, showgrid=False, showbackground=False),
        yaxis=dict(showticklabels=False, showgrid=False, showbackground=False),
        zaxis=dict(
            showticklabels=True,
            showgrid=False,
            showbackground=True,
            tickvals=[0, 3],
            ticktext=["Basse", "Haute"],
            title=dict(text="Performance", font=dict(size=14)),
            range=[0, 3],
        ),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
    ),
    width=1000,
    height=700,
)


fig.write_html(get_output_path("html"))
