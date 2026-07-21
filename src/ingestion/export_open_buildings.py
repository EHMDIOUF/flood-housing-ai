"""
Re-export Google Open Buildings footprints for both regions, keeping only
the polygon geometry and the non-geometry attributes we actually need.
The raw dataset includes a `longitude_latitude` field that is itself a
Geometry (a point), which Shapefile export cannot handle alongside the
main polygon geometry -- so we drop it here.
"""
import ee
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

DRIVE_ROOT_FOLDER = "flood_housing_ai"

KEEP_PROPERTIES = ["area_in_meters", "confidence", "full_plus_code"]


def clean_feature(feature):
    return ee.Feature(feature.geometry(), feature.toDictionary(KEEP_PROPERTIES))


submitted = []

for region_key, cfg in REGIONS.items():
    bbox = cfg["bbox"]
    geom = ee.Geometry.Rectangle(bbox)
    drive_folder = f"{DRIVE_ROOT_FOLDER}/{region_key}"

    buildings = (
        ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
        .filterBounds(geom)
        .map(clean_feature)
    )

    task = ee.batch.Export.table.toDrive(
        collection=buildings,
        description=f"{region_key}_open_buildings_v2",
        folder=drive_folder,
        fileFormat="SHP",
    )
    task.start()
    print(f"submitted: {region_key}_open_buildings_v2 (id={task.id})")
    submitted.append(task)

print(f"\nSubmitted {len(submitted)} building-footprint export tasks.")
