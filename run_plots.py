#!/usr/bin/env python3

import subprocess
import warnings
from pathlib import Path

from PIL import Image


def run_plots():
    """Run plot.py files in each subfolder and check generated image sizes."""

    src_dir = Path("src/graphes_livre")
    figures_dir = Path("figures")

    # Ensure figures directory exists
    figures_dir.mkdir(exist_ok=True)

    # Get all subdirectories in src/graphes_livre
    for subfolder in sorted(src_dir.iterdir()):
        if not subfolder.is_dir() or subfolder.name.startswith("__"):
            continue

        plot_file = subfolder / "plot.py"

        if plot_file.exists():
            print(f"Running plot.py in {subfolder.name}...")

            # Run the plot.py file
            result = subprocess.run(
                ["uv", "run", str(plot_file)],
                cwd=subfolder,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ Successfully ran {subfolder.name}/plot.py")

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
                        print(f"✓ Image {expected_image} size: {width}x{height}px")
            else:
                print(f"⚠ Expected image {expected_image} not found")
        else:
            print(f"⚠ No plot.py found in {subfolder.name}")


if __name__ == "__main__":
    run_plots()
