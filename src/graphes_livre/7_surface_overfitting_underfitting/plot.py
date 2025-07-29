import geopandas as gpd
import numpy as np
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.spatial import cKDTree
from shapely.geometry import Point
from tqdm import tqdm

from graphes_livre import get_output_path


# Generate random points and classify them as inside or outside France
def generate_random_points_and_labels(france, n_points=1000):
    # Generate random points (longitude, latitude) within a certain range
    longitudes = np.random.uniform(-5, 10, n_points)
    latitudes = np.random.uniform(41, 52, n_points)
    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(longitudes, latitudes)]
    )

    # Check if each point is inside the France polygon
    points["in_france"] = points.within(france.unary_union)

    # Convert points to tensors
    data = torch.tensor(
        [[point.x, point.y] for point in points.geometry], dtype=torch.float32
    )
    labels = torch.tensor(
        points["in_france"].astype(int).values, dtype=torch.float32
    ).unsqueeze(1)

    return points, data, labels


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


def crop_to_half_size(image_path, output_path, margin_sides=300, margin_top=200):
    """
    Crop an image to half its original size with optional margins.

    Args:
        image_path: Path to input image
        output_path: Path to save cropped image
        margin_sides: Horizontal margin from sides (positive = move inward)
        margin_top: Vertical margin from top (positive = move down)
    """
    # Open the image
    img = Image.open(image_path)

    # Get original dimensions
    width, height = img.size

    target_width = width // 2
    target_height = height // 2

    left = (width - target_width) // 2 - margin_sides
    top = (height - target_height) // 2 + margin_top

    right = left + target_width + 2 * margin_sides
    bottom = top + target_height

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    cropped_img = img.crop((left, top, right, bottom))

    # Save the result
    cropped_img.save(output_path)


def neural_net(longitude_grid, latitude_grid, model):
    grid_points = torch.tensor(
        np.c_[longitude_grid.ravel(), latitude_grid.ravel()], dtype=torch.float32
    )
    with torch.no_grad():
        grid_predictions = model(grid_points).numpy().reshape(longitude_grid.shape)
    grid_predictions = np.nan_to_num(grid_predictions, nan=1.0, posinf=1.0, neginf=0.0)
    return grid_predictions


def underfitting_gaussian(longitude, latitude):
    # Dummy prediction logic for the purpose of this illustration
    # We'll just use a simple function that vaguely represents a decision boundary
    return np.exp(-((longitude - 2) ** 2 + (latitude - 46) ** 2) / 30)


def create_nearest_neighbor_function(random_points):
    def nearest_neighbor(target_lon, target_lat):
        # Create a KDTree for fast spatial search
        tree = cKDTree(random_points[["latitude", "longitude"]].values)

        # Query the tree for the nearest point
        dist, idx = tree.query([target_lat, target_lon])

        # Return the row of the nearest neighbor
        res = random_points.iloc[idx]["in_france"]
        return res

    return np.vectorize(nearest_neighbor)


def create_overfitting_nearest_neighbor_function(random_points):
    def overfitting_nearest_neightbor(target_lon, target_lat):
        # Create a KDTree for fast spatial search
        tree = cKDTree(random_points[["latitude", "longitude"]].values)

        # Query the tree for the nearest point
        dist, idx = tree.query([target_lat, target_lon])

        # Return the row of the nearest neighbor
        if dist > 0.1:
            return False
        else:
            res = random_points.iloc[idx]["in_france"]
            return res

    return np.vectorize(overfitting_nearest_neightbor)


def main():
    N_RANDOM_POINTS = 3000

    # Load the GeoJSON file for France
    france = gpd.read_file("data/metropole-simplifiee.geojson")

    # Generate random points and labels
    random_points, random_points_data, random_points_labels = (
        generate_random_points_and_labels(france, n_points=N_RANDOM_POINTS)
    )

    random_points["longitude"] = random_points["geometry"].apply(lambda x: x.x)
    random_points["latitude"] = random_points["geometry"].apply(lambda x: x.y)

    # Define model parameters
    input_size = 2  # Longitude and Latitude
    hidden_size = 30  # Number of neurons in the hidden layer
    output_size = 1  # Output size (1 neuron for binary classification)

    # Instantiate the model
    model = SimpleNN(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    learning_rate = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 50000
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate / 10, last_epoch=-1
    )
    pbar = tqdm(range(num_epochs))
    losses = []
    for epoch in pbar:
        # Forward pass
        outputs = model(random_points_data)
        loss = criterion(outputs, random_points_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        epoch_step = 100
        if (epoch + 1) % epoch_step == 0:
            pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}"
            )
            losses.append(loss.item())

    # plt.plot(losses)

    # Set the same camera view and aspect ratio
    camera = dict(eye=dict(x=0.2, y=-2, z=1))
    aspectratio = dict(x=1, y=1, z=0.2)

    zaxis_settings = dict(
        title="Valeur réelle",
        range=[0, 1],
        tickvals=[0, 0.5, 1],  # Set specific tick values
        ticktext=["0", "0.5", "1"],  # Set tick labels
        gridcolor="lightgrey",  # Color of the grid lines
        gridwidth=2,
    )

    scene_config = dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis=zaxis_settings,
        aspectratio=aspectratio,
        camera=camera,
        xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", showbackground=False),
        yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", showbackground=False),
    )

    # First figure: Scatter3D plot
    fig1 = go.Figure()

    # Scatter points
    scatter = go.Scatter3d(
        x=[point.x for point in random_points["geometry"]],
        y=[point.y for point in random_points["geometry"]],
        z=random_points["in_france"] * 0.9 + 0.1,
        mode="markers",
        marker=dict(
            size=3,
            color=["red" if point else "blue" for point in random_points["in_france"]],
            opacity=1.0,  # Add some transparency if needed
        ),
    )

    fig1.add_trace(scatter)

    fig1.update_layout(
        scene=scene_config,
        height=700,
        width=900,
    )

    objectif_path = get_output_path("jpg").replace(".jpg", "_objectif.jpg")
    fig1.write_image(objectif_path, scale=6, width=900, height=700)
    crop_to_half_size(objectif_path, objectif_path, margin_sides=300, margin_top=200)

    # Generate grid data for latitude and longitude
    GRID_RESOLUTION = 50
    longitude = np.linspace(-5, 10, num=GRID_RESOLUTION)
    latitude = np.linspace(41, 52, num=GRID_RESOLUTION)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)

    # Create function instances
    nearest_neighbor = create_nearest_neighbor_function(random_points)
    overfitting_nearest_neightbor = create_overfitting_nearest_neighbor_function(
        random_points
    )

    scene_config["zaxis"]["title"] = "Prédiction"

    dictionnaire_fonctions = {
        "neural_net": lambda lon_grid, lat_grid: neural_net(lon_grid, lat_grid, model),
        "underfitting": underfitting_gaussian,
        "overfitting": overfitting_nearest_neightbor,
    }

    for name, function in dictionnaire_fonctions.items():
        # Second figure: Surface plot
        fig = go.Figure()

        grid_predictions = function(longitude_grid, latitude_grid)

        surface = go.Surface(
            z=np.minimum(grid_predictions, 0.95),
            x=longitude_grid,
            y=latitude_grid,
            cmin=0,
            cmax=1.1,
            opacity=1.0,  # Set opacity to 0.5 for 50% transparency
            showscale=False,
            # colorbar=dict(title="Prédiction du modèle"),
        )

        fig.add_trace(surface)

        fig.update_layout(scene=scene_config, height=700, width=900)
        surface_path = get_output_path("jpg")
        # For multiple files, we need to create unique names
        surface_path = surface_path.replace(".jpg", f"_surface_{name}.jpg")
        fig.write_image(surface_path, scale=6)
        crop_to_half_size(surface_path, surface_path, margin_sides=300, margin_top=200)


if __name__ == "__main__":
    main()
