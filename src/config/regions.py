"""
Bounding boxes for the two study regions.
These are APPROXIMATE first-pass extents based on the proposal's named
corridors -- refine by inspecting the exported imagery/footprints and
tightening these boxes in Step 5 if needed.
"""

REGIONS = {
    "region_a_kelani": {
        "label": "Lower Kelani River basin (Colombo/Gampaha, Sri Lanka)",
        "bbox": [79.83, 6.85, 80.05, 7.05],  # [minLon, minLat, maxLon, maxLat]
        "country_iso": "LKA",
    },
    "region_b_dakar": {
        "label": "Dakar northern suburbs (Pikine/Guediawaye/Keur Massar, Senegal)",
        "bbox": [-17.47, 14.68, -17.28, 14.82],
        "country_iso": "SEN",
    },
}
