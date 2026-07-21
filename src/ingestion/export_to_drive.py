"""
Phase 0 / Step 4: submit Earth Engine batch export tasks to Google Drive
for both regions. Covers: Sentinel-2 optical, Sentinel-1 SAR, DEM,
ESA WorldCover, WorldPop population, and Google Open Buildings footprints.

These are ASYNC server-side tasks -- this script only SUBMITS them.
Use src/ingestion/check_tasks.py afterward to monitor progress, and check
your Google Drive account (the one used for ee.Authenticate()) once done.
"""
import ee
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

DRIVE_ROOT_FOLDER = "flood_housing_ai"


def region_geom(bbox):
    return ee.Geometry.Rectangle(bbox)


def export_image(image, description, folder, region, scale, crs="EPSG:4326"):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
    )
    task.start()
    print(f"  submitted image task: {description} (id={task.id})")
    return task


def export_table(fc, description, folder):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=folder,
        fileFormat="SHP",
    )
    task.start()
    print(f"  submitted table task: {description} (id={task.id})")
    return task


submitted = []

for region_key, cfg in REGIONS.items():
    print(f"\n=== {region_key}: {cfg['label']} ===")
    bbox = cfg["bbox"]
    geom = region_geom(bbox)
    drive_folder = f"{DRIVE_ROOT_FOLDER}/{region_key}"

    # --- Sentinel-2 optical: cloud-filtered median composite, 4 bands + NIR ---
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate("2024-01-01", "2025-06-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .select(["B2", "B3", "B4", "B8"])
        .clip(geom)
    )
    submitted.append(export_image(s2, f"{region_key}_s2_composite", drive_folder, geom, scale=10))

    # --- Sentinel-1 SAR: VV/VH median composite (IW mode) ---
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filterDate("2024-01-01", "2025-06-30")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .median()
        .clip(geom)
    )
    submitted.append(export_image(s1, f"{region_key}_s1_composite", drive_folder, geom, scale=10))

    # --- DEM: Copernicus GLO-30 ---
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().clip(geom)
    submitted.append(export_image(dem, f"{region_key}_dem", drive_folder, geom, scale=30))

    # --- ESA WorldCover 2021 land cover ---
    worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(geom)
    submitted.append(export_image(worldcover, f"{region_key}_worldcover", drive_folder, geom, scale=10))

    # --- WorldPop population, most recent available year ---
    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", cfg["country_iso"]))
        .sort("system:time_start", False)
        .first()
        .clip(geom)
    )
    submitted.append(export_image(worldpop, f"{region_key}_worldpop", drive_folder, geom, scale=100))

    # --- Google Open Buildings footprints (vector, as Shapefile) ---
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons").filterBounds(geom)
    submitted.append(export_table(buildings, f"{region_key}_open_buildings", drive_folder))

print(f"\nSubmitted {len(submitted)} export tasks total.")
print("Run: python src/ingestion/check_tasks.py    to monitor progress.")
