"""
Exports JRC Global Human Settlement Layer (GHSL) built-up surface data
for both regions across multiple epochs, as an authoritative,
purpose-built alternative to our noisy from-scratch yearly classifier.

Dataset: JRC/GHSL/P2023A/GHS_BUILT_S -- "built_surface" band gives the
built-up surface area (m^2) per 100m grid cell, available at 5-year
epochs from 1975 to 2030 (2025/2030 are model-based projections).
We pull 2000, 2010, 2015, 2020, 2025 to cover a broad, well-supported
span up to and slightly beyond our own study period.
"""
import ee
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

DRIVE_ROOT_FOLDER = "flood_housing_ai"
EPOCHS = [2000, 2010, 2015, 2020, 2025]

submitted = []

for region_key, cfg in REGIONS.items():
    geom = ee.Geometry.Rectangle(cfg["bbox"])

    for epoch in EPOCHS:
        img = (
            ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
            .filterDate(f"{epoch}-01-01", f"{epoch}-12-31")
            .first()
            .select("built_surface")
            .clip(geom)
        )

        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"{region_key}_ghsl_{epoch}",
            folder=f"{DRIVE_ROOT_FOLDER}_{region_key}",
            region=geom,
            scale=100,  # GHSL's native resolution for this product
            crs="EPSG:4326",
            maxPixels=1e13,
        )
        task.start()
        print(f"submitted: {region_key}_ghsl_{epoch} (id={task.id})")
        submitted.append(task)

print(f"\nSubmitted {len(submitted)} GHSL export tasks.")
