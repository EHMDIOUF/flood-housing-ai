"""
Re-export Sentinel-2 composites for both regions, adding B11 (SWIR),
needed to compute NDBI. B8A (narrow NIR) is also included since it's
sometimes preferred over B8 for index calculations, and B12 for
completeness -- all are free within the same image, negligible extra cost.
"""
import ee
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

DRIVE_ROOT_FOLDER = "flood_housing_ai"
BANDS = ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]

submitted = []

for region_key, cfg in REGIONS.items():
    geom = ee.Geometry.Rectangle(cfg["bbox"])
    drive_folder = f"{DRIVE_ROOT_FOLDER}_{region_key}"  # flat name, avoids the earlier slash issue

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate("2024-01-01", "2025-06-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .select(BANDS)
        .clip(geom)
    )

    task = ee.batch.Export.image.toDrive(
        image=s2,
        description=f"{region_key}_s2_full",
        folder=drive_folder,
        region=geom,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    print(f"submitted: {region_key}_s2_full (id={task.id})")
    submitted.append(task)

print(f"\nSubmitted {len(submitted)} full-band S2 export tasks.")
