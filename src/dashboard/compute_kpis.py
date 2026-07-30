"""
Precomputes all dashboard KPIs once and caches to JSON, so the Streamlit
app itself stays fast. Includes:
  - Total built-up area (km^2) and flood-exposed built-up area (km^2)
  - % of built-up area exposed
  - GHSL growth rate (km^2/year, R^2, p-value)
  - Estimated exposed building COUNT (not just area), via building
    footprint centroids sampled against the exposure raster
"""
import rasterio
import geopandas as gpd
import numpy as np
import json
from scipy import stats
from pathlib import Path

REGIONS = {
    "region_a_kelani": "Lower Kelani Basin (Colombo/Gampaha, Sri Lanka)",
    "region_b_dakar": "Dakar Northern Suburbs (Senegal)",
}
EPOCHS = [2000, 2010, 2015, 2020, 2025]
PIXEL_AREA_KM2 = (10 * 10) / 1e6  # 10m x 10m pixels -> km^2

results = {}

for region, label in REGIONS.items():
    print(f"\n=== {region} ===")
    r = {"label": label}

    # --- Built-up / exposure area stats ---
    with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as src:
        names = list(src.descriptions)
        water = src.read(names.index("WATER_MASK") + 1)

    with rasterio.open(Path("data/processed") / f"{region}_builtup_label.tif") as src:
        builtup = src.read(1)
    with rasterio.open(Path("data/processed") / f"{region}_exposure_map.tif") as src:
        exposure = src.read(1)
    with rasterio.open(Path("data/processed") / f"{region}_flood_prone_mask.tif") as src:
        flood_prone = src.read(1)

    land = water == 0
    builtup_land = (builtup == 1) & land

    r["total_builtup_km2"] = float(builtup_land.sum() * PIXEL_AREA_KM2)
    r["exposed_builtup_km2"] = float(exposure.sum() * PIXEL_AREA_KM2)
    r["pct_builtup_exposed"] = float(100 * exposure.sum() / builtup_land.sum()) if builtup_land.sum() > 0 else 0.0
    r["pct_land_flood_prone"] = float(100 * flood_prone.sum() / land.sum())
    print(f"  Total built-up: {r['total_builtup_km2']:.2f} km^2")
    print(f"  Exposed built-up: {r['exposed_builtup_km2']:.2f} km^2 ({r['pct_builtup_exposed']:.1f}%)")

    # --- GHSL growth trend ---
    totals = []
    for epoch in EPOCHS:
        with rasterio.open(Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif") as src:
            arr = src.read(1).astype(np.float64)
            arr = np.where(arr < 0, 0, arr)
        totals.append(arr.sum() / 1e6)  # km^2

    slope, intercept, r_value, p_value, std_err = stats.linregress(EPOCHS, totals)
    r["ghsl_epochs"] = EPOCHS
    r["ghsl_builtup_km2_by_epoch"] = totals
    r["growth_rate_km2_per_year"] = float(slope)
    r["growth_r_squared"] = float(r_value ** 2)
    r["growth_p_value"] = float(p_value)
    r["total_growth_pct_2000_2025"] = float(100 * (totals[-1] - totals[0]) / totals[0])
    print(f"  Growth rate: {slope:.3f} km^2/year (R^2={r_value**2:.3f}, p={p_value:.4f})")

    # --- Estimated exposed building COUNT ---
    buildings_path = Path("data/raw") / region / f"{region}_open_buildings_v2.shp"
    gdf = gpd.read_file(buildings_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    with rasterio.open(Path("data/processed") / f"{region}_exposure_map.tif") as src:
        exposure_crs = src.crs
        exposure_transform = src.transform
        exposure_arr = src.read(1)

    gdf = gdf.to_crs(exposure_crs)
    centroids = gdf.geometry.centroid

    # Convert centroid coords to pixel row/col
    inv_transform = ~exposure_transform
    cols, rows = inv_transform * (centroids.x.values, centroids.y.values)
    cols = np.clip(cols.astype(int), 0, exposure_arr.shape[1] - 1)
    rows = np.clip(rows.astype(int), 0, exposure_arr.shape[0] - 1)

    is_exposed = exposure_arr[rows, cols] == 1
    r["total_buildings"] = int(len(gdf))
    r["estimated_exposed_buildings"] = int(is_exposed.sum())
    r["pct_buildings_exposed"] = float(100 * is_exposed.sum() / len(gdf))
    print(f"  Total buildings: {r['total_buildings']}")
    print(f"  Estimated exposed buildings: {r['estimated_exposed_buildings']} ({r['pct_buildings_exposed']:.1f}%)")

    results[region] = r

out_path = Path("data/processed") / "dashboard_kpis.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {out_path}")
