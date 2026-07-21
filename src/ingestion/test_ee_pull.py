"""
Sanity check: pull a SMALL cloud-filtered Sentinel-2 patch over the lower
Kelani basin (Region A) to confirm the ingestion chain works end-to-end.
This AOI is deliberately tiny -- full-region downloads come in Step 4.
"""
import ee
import geemap

EE_PROJECT = "fyp-flood-mapping"
ee.Initialize(project=EE_PROJECT)

# Small ~2km x 2km test patch inside the Kelani basin corridor
region_a_test = ee.Geometry.Rectangle([79.90, 6.93, 79.92, 6.95])

s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(region_a_test)
    .filterDate("2025-01-01", "2025-06-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .median()
    .select(["B2", "B3", "B4", "B8"])
    .clip(region_a_test)
)

out_path = "data/raw/region_a_kelani/s2_test_composite.tif"
geemap.ee_export_image(
    s2,
    filename=out_path,
    scale=10,
    region=region_a_test,
    file_per_band=False,
)

print(f"Saved test composite to {out_path}")
