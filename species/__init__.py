"""
Species scoring registry.

Each species module exposes:
    WEIGHTS     — dict of {feature_name: weight}
    BANDS       — list of intensity thresholds for polygon export
    score_grid(bbox, tif_path, date_str, output_dir) -> dict with
        path, grid, lats, lons, sub_scores, weights, band_count, band_mean
"""

SPECIES = {
    "blue_marlin": {
        "id": "blue_marlin",
        "label": "Blue Marlin",
        "geojson": "blue_marlin_hotspots.geojson",
        "color_ramp": "yellow-red",
        "layer_id": "hs-layer",
        "toggle_id": "hs",
    },
    "spanish_mackerel": {
        "id": "spanish_mackerel",
        "label": "Spanish Mackerel",
        "geojson": "spanish_mackerel_hotspots.geojson",
        "color_ramp": "blue-purple",
        "layer_id": "sm-hotspot-layer",
        "toggle_id": "sm-hs",
    },
    "southern_bluefin_tuna": {
        "id": "southern_bluefin_tuna",
        "label": "SB Tuna",
        "geojson": "sbt_hotspots.geojson",
        "color_ramp": "teal-cyan",
        "layer_id": "sbt-hotspot-layer",
        "toggle_id": "sbt-hs",
    },
}


def get_scorer(species_id):
    """Return the scoring function for a species."""
    if species_id == "blue_marlin":
        from marlin_data import generate_blue_marlin_hotspots
        return generate_blue_marlin_hotspots
    elif species_id == "spanish_mackerel":
        from species.spanish_mackerel import generate_spanish_mackerel_hotspots
        return generate_spanish_mackerel_hotspots
    elif species_id == "southern_bluefin_tuna":
        from species.southern_bluefin_tuna import generate_sbt_hotspots
        return generate_sbt_hotspots
    else:
        raise ValueError(f"Unknown species: {species_id}")
