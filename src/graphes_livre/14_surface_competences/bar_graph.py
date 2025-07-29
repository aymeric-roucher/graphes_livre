import matplotlib.pyplot as plt

from graphes_livre import apply_template_matplotlib, get_output_path

domaines = [
    "Calcul",
    "Logique",
    "Traduction",
    "Mémoire",
    "Spatialisation",
]
humain = [4, 4, 4, 4, 3]
aigle = [0, 0.5, 0, 3, 4]
calculatrice = [5, 0, 0, 0, 0]
llm = [2, 1, 5, 3, 1]

# Create figure and subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 12), height_ratios=[1, 1, 1, 1])
plt.subplots_adjust(hspace=0.3)  # Add space between subplots

# Colors matching your original plot
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#999999"]
data = [humain, aigle, calculatrice, llm]
labels = ["Humain", "Aigle", "Outil classique :\nCalculatrice", "LLM"]

# Create each subplot
for ax, d, color, label in zip(axes, data, colors, labels):
    # Create bars
    bars = ax.bar(domaines, d, color=color)

    # Set y-axis limits
    ax.set_ylim(0, 5)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add label on the right
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(label, rotation=0, labelpad=40, ha="left", va="center")

    # Only show x-axis ticklabels for the bottom plot
    if ax != axes[-1]:
        ax.set_xticklabels([])
    else:
        for tick in ax.get_xticklabels():
            tick.set_fontfamily("Open Sans")
            tick.set_fontweight("bold")

    # Remove y-axis ticklabels
    ax.set_yticklabels([])

# Add title and adjust layout
apply_template_matplotlib(fig, width=6, height=6, font_size=10)
plt.tight_layout()
# plt.subplots_adjust(right=0.85)
plt.savefig(get_output_path("jpg"), dpi=800, bbox_inches='tight')
