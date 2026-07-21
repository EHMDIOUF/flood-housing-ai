"""
Phase 1 preprocessing: resample all layers onto the Sentinel-2 10m grid,
compute NDVI and NDBI, and write one aligned multi-band GeoTIFF per region.

Output band order:
  1: B2 (Blue)      2: B3 (Green)     3: B4 (Red)
  4: B8 (NIR)       5: B8A            6: B11 (SWIR1)
  7: B12 (SWIR2)    8: NDVI           9: NDBI
  10: S1_VV         11: S1_VH
  12: DEM           13: WorldCover    14: WorldPop
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]


def resample_to_reference(src_path, ref_profile, resampling_method):
    """Read src_path and resample it onto the reference grid/profile."""
    with rasterio.open(src_path) as src:
        dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling_method,
        )
    return dst


for region in REGIONS:
    print(f"\n=== Processing {region} ===")
    base = Path("data/raw") / region

    # --- Reference grid: Sentinel-2 full stack ---
    s2_path = base / f"{region}_s2_full.tif"
    with rasterio.open(s2_path) as ref:
        ref_profile = ref.profile.copy()
        s2_bands = ref.read().astype(np.float32)  # shape (7, H, W)
        band_names = list(ref.descriptions)

    B2, B3, B4, B8, B8A, B11, B12 = s2_bands

    # --- Spectral indices ---
    eps = 1e-6
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndbi = (B11 - B8) / (B11 + B8 + eps)

    # --- Resample Sentinel-1 (bilinear: continuous backscatter) ---
    s1_path = base / f"{region}_s1_composite.tif"
    with rasterio.open(s1_path) as src:
        s1_vv = resample_to_reference(s1_path, ref_profile, Resampling.bilinear) if src.count >= 1 else None
    with rasterio.open(s1_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32) if src.count >= 2 else None
    # Resample VV and VH bands individually
    def resample_band(path, band_idx, method):
        with rasterio.open(path) as src:
            dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=method,
            )
        return dst

    s1_vv = resample_band(s1_path, 1, Resampling.bilinear)
    s1_vh = resample_band(s1_path, 2, Resampling.bilinear)

    # --- Resample DEM (bilinear: continuous elevation) ---
    dem_path = base / f"{region}_dem.tif"
    dem = resample_band(dem_path, 1, Resampling.bilinear)

    # --- Resample WorldCover (nearest: categorical land-cover classes) ---
    wc_path = base / f"{region}_worldcover.tif"
    worldcover = resample_band(wc_path, 1, Resampling.nearest)

    # --- Resample WorldPop (bilinear: continuous population density) ---
    wp_path = base / f"{region}_worldpop.tif"
    worldpop = resample_band(wp_path, 1, Resampling.bilinear)

    # --- Stack everything ---
    stack = np.stack(
        [B2, B3, B4, B8, B8A, B11, B12, ndvi, ndbi, s1_vv, s1_vh, dem, worldcover, worldpop],
        axis=0,
    )
    out_band_names = [
        "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
        "NDVI", "NDBI", "S1_VV", "S1_VH", "DEM", "WorldCover", "WorldPop",
    ]

    out_profile = ref_profile.copy()
    out_profile.update(count=stack.shape[0], dtype="float32", compress="lzw")

    out_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stack)
        for i, name in enumerate(out_band_names, start=1):
            dst.set_band_description(i, name)

    print(f"Saved: {out_path}  shape={stack.shape}  bands={out_band_names}")
