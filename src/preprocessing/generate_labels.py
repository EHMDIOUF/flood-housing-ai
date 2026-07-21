"""
Generates a binary built-up label raster (1=built-up, 0=not) aligned to
the feature stack grid, combining:
  - ESA WorldCover class 50 (built-up) as the base layer
  - Google Open Buildings footprints, rasterized and OR'd in, since
    footprints are generally more spatially precise than WorldCover's
    10m land-cover classes for pinpointing actual structures
"""
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
WORLDCOVER_BUILTUP_CLASS = 50

for region in REGIONS:
    print(f"\n=== {region} ===")
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    buildings_path = Path("data/raw") / region / f"{region}_open_buildings_v2.shp"

    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        names = list(src.descriptions)
        transform = src.transform
        shape = (src.height, src.width)
        worldcover = src.read(names.index("WorldCover") + 1)

    # Base label from WorldCover
    label_from_wc = (worldcover == WORLDCOVER_BUILTUP_CLASS).astype(np.uint8)
    print(f"  WorldCover built-up pixels: {label_from_wc.sum()} ({100*label_from_wc.mean():.2f}%)")

    # Rasterize building footprints
    gdf = gpd.read_file(buildings_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(profile["crs"])

    label_from_buildings = rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    print(f"  Building-footprint pixels: {label_from_buildings.sum()} ({100*label_from_buildings.mean():.2f}%)")

    # Combine: built-up if EITHER source says so
    combined_label = np.maximum(label_from_wc, label_from_buildings)
    print(f"  Combined built-up pixels: {combined_label.sum()} ({100*combined_label.mean():.2f}%)")

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="lzw")
    out_path = Path("data/processed") / f"{region}_builtup_label.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(combined_label[np.newaxis, :, :])
        dst.set_band_description(1, "BUILTUP_LABEL")

    print(f"  saved: {out_path}")
