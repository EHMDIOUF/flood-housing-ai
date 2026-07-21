"""
Exports Sentinel-2 cloud-filtered median composites for 4 time points
(2019, 2021, 2023, 2025) for both regions, for expansion-trend analysis.
Each snapshot is a 6-month window (Jan-Jun) centered similarly to our
existing composite, to keep seasonal conditions comparable across years.
Bands: B4, B3, B2, B8, B11 (RGB + NIR + SWIR -- enough for NDVI/NDBI
per-year without re-pulling all 7 bands, keeping exports lighter).
"""
import ee
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

DRIVE_ROOT_FOLDER = "flood_housing_ai"
YEARS = [2019, 2021, 2023, 2025]
BANDS = ["B2", "B3", "B4", "B8", "B11"]

submitted = []

for region_key, cfg in REGIONS.items():
    geom = ee.Geometry.Rectangle(cfg["bbox"])

    for year in YEARS:
        start = f"{year}-01-01"
        end = f"{year}-06-30"

        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .select(BANDS)
            .clip(geom)
        )

        # Report how many source images went into this composite (sanity check for data availability)
        count = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .size()
            .getInfo()
        )
        print(f"{region_key} {year}: {count} source images in composite")

        task = ee.batch.Export.image.toDrive(
            image=s2,
            description=f"{region_key}_s2_{year}",
            folder=f"{DRIVE_ROOT_FOLDER}_{region_key}",
            region=geom,
            scale=10,
            crs="EPSG:4326",
            maxPixels=1e13,
        )
        task.start()
        print(f"  submitted: {region_key}_s2_{year} (id={task.id})")
        submitted.append(task)

print(f"\nSubmitted {len(submitted)} temporal export tasks.")
