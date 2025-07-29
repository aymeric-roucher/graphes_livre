#!/usr/bin/env python3

import subprocess
import warnings
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def run_plots():
    """Run plot.py files in each subfolder and check generated image sizes."""

    src_dir = Path("src/graphes_livre")
    figures_dir = Path("figures")

    # Ensure figures directory exists
    figures_dir.mkdir(exist_ok=True)

    print("Running plots...")

    # Get all subdirectories in src/graphes_livre
    for subfolder in tqdm(sorted(src_dir.iterdir())):
        if not subfolder.is_dir() or subfolder.name.startswith("__"):
            continue

        plot_file = subfolder / "plot.py"

        if plot_file.exists():
            # Run the plot.py file
            subprocess.run(
                ["uv", "run", str(plot_file)],
                # Do not set cwd, so the script runs in the current working directory
                check=True,
                capture_output=True,
                text=True,
            )

            # Check generated image size
            expected_image = figures_dir / f"{subfolder.name}.jpg"

            if expected_image.exists():
                with Image.open(expected_image) as img:
                    width, height = img.size

                    if width < 1000 or height < 1000:
                        warnings.warn(
                            f"Image {expected_image} has size {width}x{height}px, which is below 1000x1000px"
                        )
            else:
                raise ValueError(f"⚠ Expected image {expected_image} not found")
        else:
            print(f"⚠ No plot.py found in {subfolder.name}, skipping it.")


if __name__ == "__main__":
    run_plots()
