"""
QA pass over all raw downloaded layers for both regions:
- Reports CRS, resolution, dimensions, bounds for each raster
- Reports CRS, feature count, bounds for the buildings shapefile
- Flags CRS mismatches and non-overlapping extents within each region
"""
import rasterio
import geopandas as gpd
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
RASTER_LAYERS = ["s1_composite", "s2_composite", "dem", "worldcover", "worldpop"]

for region in REGIONS:
    print(f"\n{'='*70}\n{region}\n{'='*70}")
    base = Path("data/raw") / region

    raster_bounds = {}
    raster_crs = {}

    for layer in RASTER_LAYERS:
        path = base / f"{region}_{layer}.tif"
        if not path.exists():
            print(f"  [MISSING] {path.name}")
            continue
        with rasterio.open(path) as src:
            raster_bounds[layer] = src.bounds
            raster_crs[layer] = src.crs
            print(f"  {layer:15s} CRS={str(src.crs):15s} size={src.width}x{src.height:<6} "
                  f"res={src.res} bounds={tuple(round(b,4) for b in src.bounds)}")

    shp_path = base / f"{region}_open_buildings_v2.shp"
    if shp_path.exists():
        gdf = gpd.read_file(shp_path)
        print(f"  {'buildings':15s} CRS={str(gdf.crs):15s} features={len(gdf):<6} "
              f"bounds={tuple(round(b,4) for b in gdf.total_bounds)}")
    else:
        print(f"  [MISSING] {shp_path.name}")

    # Flag CRS mismatches
    crs_values = set(str(c) for c in raster_crs.values())
    if len(crs_values) > 1:
        print(f"  [WARNING] Mixed CRS across rasters: {raster_crs}")
    else:
        print(f"  [OK] All rasters share CRS: {crs_values}")

    # Flag non-overlapping bounds (rough check: do all bounding boxes intersect?)
    boxes = list(raster_bounds.values())
    if boxes:
        min_left = max(b.left for b in boxes)
        max_right = min(b.right for b in boxes)
        min_bottom = max(b.bottom for b in boxes)
        max_top = min(b.top for b in boxes)
        if min_left < max_right and min_bottom < max_top:
            print(f"  [OK] All raster extents overlap.")
        else:
            print(f"  [WARNING] Raster extents do NOT all overlap -- check bounds above.")
