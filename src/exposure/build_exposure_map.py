"""
Builds the flood exposure map for both regions:
  1. Flood-prone mask: HAND < 5m (literature-standard threshold)
  2. Exposure = flood-prone AND built-up (from Phase 1's combined label)
Reports area statistics and saves both the flood-prone mask and the
final exposure raster.
"""
import rasterio
import numpy as np
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
HAND_THRESHOLD_M = 5.0

for region in REGIONS:
    print(f"\n=== {region} ===")
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    label_path = Path("data/processed") / f"{region}_builtup_label.tif"

    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        names = list(src.descriptions)
        hand = src.read(names.index("HAND") + 1)
        water_mask = src.read(names.index("WATER_MASK") + 1)

    with rasterio.open(label_path) as src:
        builtup = src.read(1)

    # --- Flood-prone mask (exclude water pixels -- flood-proneness of open water is meaningless) ---
    flood_prone = (hand < HAND_THRESHOLD_M).astype(np.uint8)
    flood_prone[water_mask == 1] = 0

    land_pixels = (water_mask == 0).sum()
    flood_prone_pixels = flood_prone.sum()
    print(f"  Flood-prone (HAND < {HAND_THRESHOLD_M}m) land pixels: {flood_prone_pixels} "
          f"({100*flood_prone_pixels/land_pixels:.2f}% of land area)")

    # --- Exposure: flood-prone AND built-up ---
    exposure = ((flood_prone == 1) & (builtup == 1)).astype(np.uint8)
    builtup_land_pixels = ((builtup == 1) & (water_mask == 0)).sum()
    exposure_pixels = exposure.sum()

    print(f"  Built-up land pixels: {builtup_land_pixels}")
    print(f"  Exposed (flood-prone + built-up) pixels: {exposure_pixels} "
          f"({100*exposure_pixels/builtup_land_pixels:.2f}% of built-up area is flood-exposed)")

    # --- Save flood-prone mask ---
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", compress="lzw")

    fp_path = Path("data/processed") / f"{region}_flood_prone_mask.tif"
    with rasterio.open(fp_path, "w", **out_profile) as dst:
        dst.write(flood_prone[np.newaxis, :, :])
        dst.set_band_description(1, "FLOOD_PRONE")
    print(f"  Saved: {fp_path}")

    # --- Save exposure map ---
    exp_path = Path("data/processed") / f"{region}_exposure_map.tif"
    with rasterio.open(exp_path, "w", **out_profile) as dst:
        dst.write(exposure[np.newaxis, :, :])
        dst.set_band_description(1, "EXPOSED")
    print(f"  Saved: {exp_path}")
