"""
Fixes cross-year radiometric drift by percentile-matching each year's
bands to the 2025 reference composite before computing NDVI/NDBI and
classifying. This removes atmospheric/seasonal brightness differences
that were causing spurious "built-up" swings unrelated to real change.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import joblib
import gc
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
YEARS = [2019, 2021, 2023, 2025]
REFERENCE_YEAR = 2025
REDUCED_FEATURES = ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDBI"]
RAW_BANDS = ["B2", "B3", "B4", "B8", "B11"]

print("Loading reduced-feature classifier...")
clf = joblib.load("models/rf_builtup_reduced_features.pkl")

for region in REGIONS:
    ref_stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(ref_stack_path) as ref:
        ref_profile = ref.profile.copy()
    ref_h, ref_w = ref_profile["height"], ref_profile["width"]

    print(f"\n=== {region} ===")

    # --- Establish reference percentiles from the REFERENCE_YEAR composite ---
    ref_snap_path = Path("data/raw") / region / f"{region}_s2_{REFERENCE_YEAR}.tif"
    with rasterio.open(ref_snap_path) as src:
        ref_bands = [src.read(i + 1).astype(np.float32) for i in range(5)]
    ref_percentiles = {b: np.percentile(arr, [2, 98]) for b, arr in zip(RAW_BANDS, ref_bands)}
    print(f"  Reference ({REFERENCE_YEAR}) percentiles: " +
          "  ".join(f"{b}=[{p[0]:.0f},{p[1]:.0f}]" for b, p in ref_percentiles.items()))
    del ref_bands
    gc.collect()

    for year in YEARS:
        print(f"  Processing {year}...")
        snap_path = Path("data/raw") / region / f"{region}_s2_{year}.tif"

        with rasterio.open(snap_path) as src:
            src_transform, src_crs = src.transform, src.crs
            raw = {b: src.read(i + 1).astype(np.float32) for i, b in enumerate(RAW_BANDS)}

        # --- Percentile-match each band to the reference year ---
        normalized = {}
        for b in RAW_BANDS:
            lo_src, hi_src = np.percentile(raw[b], [2, 98])
            lo_ref, hi_ref = ref_percentiles[b]
            scale = (hi_ref - lo_ref) / (hi_src - lo_src + 1e-6)
            normalized[b] = (raw[b] - lo_src) * scale + lo_ref
        del raw
        gc.collect()

        eps = 1e-6
        ndvi = (normalized["B8"] - normalized["B4"]) / (normalized["B8"] + normalized["B4"] + eps)
        ndbi = (normalized["B11"] - normalized["B8"]) / (normalized["B11"] + normalized["B8"] + eps)

        resampled = np.empty((len(REDUCED_FEATURES), ref_h, ref_w), dtype=np.float32)
        band_list = [normalized["B2"], normalized["B3"], normalized["B4"],
                     normalized["B8"], normalized["B11"], ndvi, ndbi]
        for i, band_arr in enumerate(band_list):
            reproject(
                source=band_arr, destination=resampled[i],
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                resampling=Resampling.bilinear,
            )
        del normalized, ndvi, ndbi, band_list
        gc.collect()

        X_yr = resampled.reshape(len(REDUCED_FEATURES), -1).T
        del resampled
        gc.collect()

        pred_yr = clf.predict(X_yr).reshape(ref_h, ref_w).astype(np.uint8)
        del X_yr
        gc.collect()

        out_path = Path("data/processed") / f"{region}_builtup_{year}_norm.tif"
        out_profile = ref_profile.copy()
        out_profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(pred_yr[np.newaxis, :, :])
            dst.set_band_description(1, f"BUILTUP_{year}_NORM")

        pct = 100 * pred_yr.mean()
        print(f"    built-up = {pct:.2f}% of full extent -> saved {out_path}")

        del pred_yr
        gc.collect()
