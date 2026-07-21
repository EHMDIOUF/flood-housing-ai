"""
Memory-conservative version: applies the already-trained reduced-feature
classifier (models/rf_builtup_reduced_features.pkl) to each year's
snapshot, one year at a time, explicitly freeing memory between
iterations to avoid OOM on memory-constrained machines.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import joblib
import gc
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
YEARS = [2019, 2021, 2023, 2025]
REDUCED_FEATURES = ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDBI"]

print("Loading reduced-feature classifier...")
clf = joblib.load("models/rf_builtup_reduced_features.pkl")

for region in REGIONS:
    ref_stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(ref_stack_path) as ref:
        ref_profile = ref.profile.copy()
    ref_h, ref_w = ref_profile["height"], ref_profile["width"]

    print(f"\n=== {region} ===")
    for year in YEARS:
        print(f"  Processing {year}...")
        snap_path = Path("data/raw") / region / f"{region}_s2_{year}.tif"

        with rasterio.open(snap_path) as src:
            src_transform, src_crs = src.transform, src.crs
            b2 = src.read(1).astype(np.float32)
            b3 = src.read(2).astype(np.float32)
            b4 = src.read(3).astype(np.float32)
            b8 = src.read(4).astype(np.float32)
            b11 = src.read(5).astype(np.float32)

        eps = 1e-6
        ndvi = (b8 - b4) / (b8 + b4 + eps)
        ndbi = (b11 - b8) / (b11 + b8 + eps)

        # Resample each band individually, write directly into a preallocated array
        resampled = np.empty((len(REDUCED_FEATURES), ref_h, ref_w), dtype=np.float32)
        for i, band_arr in enumerate([b2, b3, b4, b8, b11, ndvi, ndbi]):
            reproject(
                source=band_arr, destination=resampled[i],
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                resampling=Resampling.bilinear,
            )

        del b2, b3, b4, b8, b11, ndvi, ndbi
        gc.collect()

        X_yr = resampled.reshape(len(REDUCED_FEATURES), -1).T
        del resampled
        gc.collect()

        pred_yr = clf.predict(X_yr).reshape(ref_h, ref_w).astype(np.uint8)
        del X_yr
        gc.collect()

        out_path = Path("data/processed") / f"{region}_builtup_{year}.tif"
        out_profile = ref_profile.copy()
        out_profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(pred_yr[np.newaxis, :, :])
            dst.set_band_description(1, f"BUILTUP_{year}")

        pct = 100 * pred_yr.mean()
        print(f"    built-up = {pct:.2f}% of full extent -> saved {out_path}")

        del pred_yr
        gc.collect()
