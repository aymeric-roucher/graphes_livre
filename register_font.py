import os

import matplotlib.font_manager as fm

# Register all Open Sans variants
open_sans_files = [
    "./OpenSans/OpenSans-Regular.ttf",
    "./OpenSans/OpenSans-Bold.ttf",
    "./OpenSans/OpenSans-Italic.ttf",
    "./OpenSans/OpenSans-BoldItalic.ttf",
    # Add other variants as needed
]

for font_file in open_sans_files:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)
        print(f"Registered: {font_file}")
    else:
        print(f"File not found: {font_file}")

# Clear cache to ensure fonts are available
fm._load_fontmanager(try_read_cache=False)
