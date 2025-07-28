import os
import sys

# Add current directory to path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

from utils import (
    BOLD_FONT_FAMILY,
    FONT_FAMILY,
    SEA_BLUE,
    apply_template,
    apply_template_matplotlib,
    get_output_path,
)

__all__ = [
    "apply_template",
    "apply_template_matplotlib",
    "FONT_FAMILY",
    "BOLD_FONT_FAMILY",
    "SEA_BLUE",
    "get_output_path",
]
