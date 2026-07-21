"""
For each region and year, computes NDVI/NDBI from the temporal Sentinel-2
snapshot and applies the trained built-up classifier (production RF from
Phase 1) to produce a per-year built-up map, resampled onto the same
10m grid as the main feature stack for direct spatial comparison.

Note: temporal snapshots only have bands [B2,B3,B4,B8,B11] (no B8A/B12/S1/DEM),
so we retrain a REDUCED-FEATURE classifier using only the features available
across all years, to keep per-year predictions apples-to-apples.
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from pathlib import Path
import joblib

REGIONS = ["region_a_kelani", "region_b_dakar"]
YEARS = [2019, 2021, 2023, 2025]
REDUCED_FEATURES = ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDBI"]
TILE_SIZE = 64


def checkerboard_mask(H, W, tile_size):
    row_tile = (np.arange(H) // tile_size)[:, None]
    col_tile = (np.arange(W) // tile_size)[None, :]
    return ((row_tile + col_tile) % 2 == 0)


def compute_reduced_features(bands_dict):
    """bands_dict: {'B2':arr, 'B3':arr, 'B4':arr, 'B8':arr, 'B11':arr}"""
    eps = 1e-6
    ndvi = (bands_dict["B8"] - bands_dict["B4"]) / (bands_dict["B8"] + bands_dict["B4"] + eps)
    ndbi = (bands_dict["B11"] - bands_dict["B8"]) / (bands_dict["B11"] + bands_dict["B8"] + eps)
    return np.stack([bands_dict["B2"], bands_dict["B3"], bands_dict["B4"],
                      bands_dict["B8"], bands_dict["B11"], ndvi, ndbi], axis=0)


# --- Step A: retrain a reduced-feature classifier on Region A (2025-equivalent bands only) ---
print("Retraining reduced-feature classifier (for cross-year consistency)...")
stack_path = Path("data/processed") / "region_a_kelani_feature_stack.tif"
label_path = Path("data/processed") / "region_a_kelani_builtup_label.tif"

with rasterio.open(stack_path) as src:
    names = list(src.descriptions)
    full_bands = src.read()
    water = full_bands[names.index("WATER_MASK")]

bands_dict = {b: full_bands[names.index(b)] for b in ["B2", "B3", "B4", "B8", "B11"]}
reduced_stack = compute_reduced_features(bands_dict)

with rasterio.open(label_path) as src:
    label = src.read(1)

H, W = label.shape
train_mask = checkerboard_mask(H, W, TILE_SIZE)

X_flat = reduced_stack.reshape(len(REDUCED_FEATURES), -1).T
y_flat = label.reshape(-1)
water_flat = water.reshape(-1)
valid_train = (water_flat == 0) & train_mask.reshape(-1)
valid_test = (water_flat == 0) & (~train_mask).reshape(-1)

clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
                              n_jobs=-1, random_state=42, class_weight="balanced")
clf.fit(X_flat[valid_train], y_flat[valid_train])

pred_test = clf.predict(X_flat[valid_test])
print(f"  Reduced-feature classifier check -- Region A held-out balanced acc: "
      f"{balanced_accuracy_score(y_flat[valid_test], pred_test):.4f} "
      f"(full-feature model was 0.8817, some drop expected from fewer bands)")

joblib.dump(clf, "models/rf_builtup_reduced_features.pkl")

# --- Step B: apply to each year's snapshot for both regions ---
for region in REGIONS:
    ref_stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(ref_stack_path) as ref:
        ref_profile = ref.profile.copy()

    print(f"\n=== {region} ===")
    for year in YEARS:
        snap_path = Path("data/raw") / region / f"{region}_s2_{year}.tif"
        with rasterio.open(snap_path) as src:
            snap_bands = src.read().astype(np.float32)  # order: B2,B3,B4,B8,B11
            src_transform, src_crs = src.transform, src.crs

        bands_dict_yr = {name: snap_bands[i] for i, name in enumerate(["B2", "B3", "B4", "B8", "B11"])}
        reduced_yr = compute_reduced_features(bands_dict_yr)

        # Resample onto reference 10m grid
        resampled = np.empty((len(REDUCED_FEATURES), ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        for i in range(len(REDUCED_FEATURES)):
            reproject(
                source=reduced_yr[i], destination=resampled[i],
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                resampling=Resampling.bilinear,
            )

        X_yr = resampled.reshape(len(REDUCED_FEATURES), -1).T
        pred_yr = clf.predict(X_yr).reshape(ref_profile["height"], ref_profile["width"]).astype(np.uint8)

        out_path = Path("data/processed") / f"{region}_builtup_{year}.tif"
        out_profile = ref_profile.copy()
        out_profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(pred_yr[np.newaxis, :, :])
            dst.set_band_description(1, f"BUILTUP_{year}")

        pct = 100 * pred_yr.mean()
        print(f"  {year}: built-up = {pct:.2f}% of full extent  -> saved {out_path}")
