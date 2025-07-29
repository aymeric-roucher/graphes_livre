import plotly.graph_objects as go
from pycountry import countries

from graphes_livre import get_output_path


def main():
    tier_1_countries = [
        "Australia",
        "Belgium",
        "Canada",
        "Denmark",
        "Finland",
        "France",
        "Germany",
        "Ireland",
        "Italy",
        "Japan",
        "Netherlands",
        "New Zealand",
        "Norway",
        "Korea, Republic of",
        "Spain",
        "Sweden",
        "Taiwan",
        "United Kingdom",
        "United States",
    ]

    # Tier 3 Countries (Highest Restrictions/Prohibited) - Country Group D:5 and Macau
    # Subject to presumption of denial for AI technologies
    tier_3_countries = [
        "Afghanistan",
        "Belarus",
        "Myanmar",
        "Cambodia",
        "Central African Republic",
        "China",  # People's Republic of China
        "Congo (Democratic Republic of)",
        "Cuba",
        "Eritrea",
        "Haiti",
        "Iran",
        "Iraq",
        "Korea, Democratic People's Republic of",  # North Korea
        "Lebanon",
        "Libya",
        "Macau",
        "Nicaragua",
        "Russian Federation",
        "Somalia",
        "South Sudan",
        "Sudan",
        "Syria",
        "Venezuela",
        "Zimbabwe",
    ]

    countries_data = []
    for country in countries:
        country.name = country.name.replace(
            "Venezuela, Bolivarian Republic of", "Venezuela"
        )
        if country.name in tier_1_countries:
            countries_data.append(
                {
                    "country": country.name,
                    "tier": "Tier 1 (Most permissive)",
                    "color_code": 1,
                }
            )
        elif country.name in tier_3_countries:
            countries_data.append(
                {
                    "country": country.name,
                    "tier": "Tier 3 (Most restrictive)",
                    "color_code": 3,
                }
            )
        else:
            if "eryth" in country.name.lower():
                print(country.name)
            countries_data.append(
                {"country": country.name, "tier": "Tier 2", "color_code": 2}
            )
    countries_data.append({"country": "Turkey", "tier": "Tier 2", "color_code": 2})
    countries_data.append({"country": "Somalia", "tier": "Tier 3", "color_code": 2})

    # Convert to lists for plotly
    countries_list = [item["country"] for item in countries_data]
    tiers = [item["tier"] for item in countries_data]
    color_codes = [item["color_code"] for item in countries_data]

    # Define custom color scale matching the original map
    color_scale = [
        [0, "#2E86AB"],  # Blue for Tier 1
        [0.5, "#F18F01"],  # Yellow/Orange for Tier 2
        [1, "#E91E63"],  # Pink/Magenta for Tier 3
    ]

    # Create the choropleth map
    fig = go.Figure(
        data=go.Choropleth(
            locations=countries_list,
            z=color_codes,
            locationmode="country names",
            colorscale=color_scale,
            showscale=False,  # Hide the default color bar since we'll add custom legend
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate="<b>%{location}</b><br>%{customdata}<extra></extra>",
            customdata=tiers,
        )
    )

    # Update layout to match the original styling
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgray",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="white",
            projection_type="natural earth",
        ),
        width=1200,
        height=700,
        margin=dict(l=20, r=20, t=100, b=100),
    )

    annotation_x, annotation_y = 0.05, 0.20
    font_size = 16
    # Add custom legend
    fig.add_annotation(
        x=annotation_x,
        y=annotation_y + 0.1,
        xref="paper",
        yref="paper",
        text="<b>■</b> Tier 1 (Export libre)",
        showarrow=False,
        font=dict(size=font_size, color="#2E86AB"),
        align="left",
        bgcolor="white",
    )

    fig.add_annotation(
        x=annotation_x,
        y=annotation_y + 0.05,
        xref="paper",
        yref="paper",
        text="<b>■</b> Tier 2",
        showarrow=False,
        font=dict(size=font_size, color="#F18F01"),
        align="left",
        bgcolor="white",
    )

    fig.add_annotation(
        x=annotation_x,
        y=annotation_y,
        xref="paper",
        yref="paper",
        text="<b>■</b> Tier 3 (Export restreint)",
        showarrow=False,
        font=dict(size=font_size, color="#E91E63"),
        align="left",
        bgcolor="white",
    )

    # Show the map
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(get_output_path("jpg"), width=900, height=600, scale=4)


if __name__ == "__main__":
    main()
