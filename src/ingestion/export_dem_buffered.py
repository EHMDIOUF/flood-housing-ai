"""
Re-exports the DEM for both regions with a spatial buffer, so flow
routing has upstream/downstream context beyond our original crop
(avoids the D8-flow-exits-tile-boundary NaN artifact in HAND).
"""
import ee
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

DRIVE_ROOT_FOLDER = "flood_housing_ai"
BUFFER_DEG = 0.05  # ~5.5km at these latitudes

submitted = []

for region_key, cfg in REGIONS.items():
    minlon, minlat, maxlon, maxlat = cfg["bbox"]
    buffered_bbox = [
        minlon - BUFFER_DEG, minlat - BUFFER_DEG,
        maxlon + BUFFER_DEG, maxlat + BUFFER_DEG,
    ]
    geom = ee.Geometry.Rectangle(buffered_bbox)

    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().clip(geom)

    task = ee.batch.Export.image.toDrive(
        image=dem,
        description=f"{region_key}_dem_buffered",
        folder=f"{DRIVE_ROOT_FOLDER}_{region_key}",
        region=geom,
        scale=30,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    print(f"submitted: {region_key}_dem_buffered (id={task.id})  bbox={buffered_bbox}")
    submitted.append(task)

print(f"\nSubmitted {len(submitted)} buffered DEM export tasks.")
