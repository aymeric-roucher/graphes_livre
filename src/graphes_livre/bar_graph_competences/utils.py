import matplotlib.pyplot as plt

FONT_FAMILY = "./OpenSans/OpenSans-Regular.ttf"
BOLD_FONT_FAMILY = "./OpenSans/OpenSans-Bold.ttf"

SEA_BLUE = "rgba(0,100,255,0.7)"


# Register all Open Sans variants
# open_sans_files = [
#     "./OpenSans/OpenSans-Regular.ttf",
#     "./OpenSans/OpenSans-Bold.ttf",
#     "./OpenSans/OpenSans-Italic.ttf",
#     "./OpenSans/OpenSans-BoldItalic.ttf",
#     # Add other variants as needed
# ]

# for font_file in open_sans_files:
#     if os.path.exists(font_file):
#         fm.fontManager.addfont(font_file)
#         print(f"Registered: {font_file}")
#     else:
#         print(f"File not found: {font_file}")


def apply_template(
    fig,
    template="none",
    annotation_text="",
    title=None,
    width=600,
    height=500,
    font_size=14,
):
    """Applies template in-place to input fig."""
    layout_updates = {
        "template": template,
        "width": width,
        "height": height,
        "font": dict(family=FONT_FAMILY, size=font_size),
        "title_font_family": BOLD_FONT_FAMILY,
        "title_font_size": 24,
        "title_xanchor": "center",
        "title_font_weight": "bold",
        "legend": dict(
            itemsizing="constant",
            title_font_family=BOLD_FONT_FAMILY,
            font=dict(family=BOLD_FONT_FAMILY, size=font_size),
            itemwidth=30,
        ),
    }
    if len(annotation_text) > 0:
        layout_updates["annotations"] = [
            dict(
                text=f"<i>{annotation_text}</i>",
                xref="paper",
                yref="paper",
                x=1.05,
                y=-0.05,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=dict(size=font_size),
            )
        ]
    if title is not None:
        layout_updates["title"] = title
    fig.update_layout(layout_updates)
    fig.update_xaxes(
        title_font_family=FONT_FAMILY,
        tickfont_family=FONT_FAMILY,
        tickfont_size=font_size,
        linewidth=1,
    )
    fig.update_yaxes(
        title_font_family=FONT_FAMILY,
        tickfont_family=FONT_FAMILY,
        tickfont_size=font_size,
        linewidth=1,
    )
    return


def apply_template_matplotlib(
    fig,
    template="none",
    annotation_text="",
    title=None,
    width=9,  # in inches
    height=5,  # in inches
    font_size=14,
):
    """Applies template in-place to input fig."""
    # Set figure size
    fig.set_size_inches(width, height)

    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.family"] = ["serif"]
    plt.rcParams["font.serif"] = ["Georgia"]

    # Apply to all axes in the figure
    for ax in fig.axes:
        # Update font for tick labels
        ax.tick_params(labelsize=font_size)

        # Update font for axis labels
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontfamily=FONT_FAMILY, fontsize=font_size)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontfamily=FONT_FAMILY, fontsize=font_size)

    if title is not None:
        fig.suptitle(title, fontfamily=FONT_FAMILY, fontsize=24, y=0.95)

    if annotation_text:
        fig.text(
            1.05,
            -0.05,
            annotation_text,
            fontfamily=FONT_FAMILY,
            fontsize=font_size,
            style="italic",
            horizontalalignment="left",
            verticalalignment="top",
            transform=fig.transFigure,
        )

    fig.tight_layout()

    return fig
