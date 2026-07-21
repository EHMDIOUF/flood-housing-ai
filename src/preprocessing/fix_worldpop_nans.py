"""
Fixes WorldPop NaN gaps in the feature stacks:
  - Water pixels (WorldCover class 80): fill with 0 (genuinely uninhabited)
  - Non-water NaN pixels: fill via nearest-neighbor spatial interpolation
    from valid WorldPop pixels (real gaps in WorldPop coverage, not water)
Adds a new WATER_MASK band (1=water, 0=land) so downstream steps can
distinguish "confirmed zero population" from "imputed population".
"""
import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
WATER_CLASS = 80

for region in REGIONS:
    path = Path("data/processed") / f"{region}_feature_stack.tif"
    print(f"\n=== {region} ===")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        names = list(src.descriptions)
        bands = src.read()

    wc_idx = names.index("WorldCover")
    wp_idx = names.index("WorldPop")

    worldcover = bands[wc_idx]
    worldpop = bands[wp_idx].copy()

    water_mask = (worldcover == WATER_CLASS).astype(np.float32)
    nan_mask = np.isnan(worldpop)

    # 1. Water NaNs -> 0
    water_nan = nan_mask & (water_mask == 1)
    worldpop[water_nan] = 0.0
    print(f"  filled {water_nan.sum()} water-NaN pixels with 0")

    # 2. Remaining (land) NaNs -> nearest-neighbor fill from valid pixels
    land_nan = nan_mask & (water_mask == 0)
    if land_nan.sum() > 0:
        valid = ~np.isnan(worldpop)
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = worldpop[tuple(idx)]
        worldpop[land_nan] = filled[land_nan]
        print(f"  interpolated {land_nan.sum()} land-NaN pixels from nearest valid neighbor")

    # Sanity: no NaNs left
    remaining = np.isnan(worldpop).sum()
    print(f"  remaining NaNs after fix: {remaining}")

    bands[wp_idx] = worldpop

    # Append water mask as a new band
    new_bands = np.concatenate([bands, water_mask[np.newaxis, :, :]], axis=0)
    new_names = names + ["WATER_MASK"]

    profile.update(count=new_bands.shape[0])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(new_bands)
        for i, name in enumerate(new_names, start=1):
            dst.set_band_description(i, name)

    print(f"  saved updated stack: {new_bands.shape[0]} bands -> {new_names}")
