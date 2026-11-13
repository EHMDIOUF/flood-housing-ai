"""
Analyzes GHSL built-up surface time series for both regions:
  1. Computes total built-up surface (m^2) per epoch, fits a linear trend
     (with R^2/p-value, since Phase 4's earlier attempt taught us to be
     careful about claiming trends without checking significance).
  2. Resamples each epoch's GHSL layer onto the 10m feature-stack grid,
     and cross-references built-up GROWTH (change between first and last
     epoch) against the Phase 2 flood-prone mask: is growth concentrated
     in flood-prone or flood-safe terrain?
"""
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from scipy import stats
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
EPOCHS = [2000, 2010, 2015, 2020, 2025]

for region in REGIONS:
    print(f"\n{'='*70}\n{region}\n{'='*70}")

    # --- Step 1: total built-up surface per epoch + trend fit ---
    totals = []
    for epoch in EPOCHS:
        path = Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif"
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            arr = np.where(arr < 0, 0, arr)  # guard against any negative/nodata sentinel
        total_m2 = arr.sum()
        totals.append(total_m2)
        print(f"  {epoch}: total built-up surface = {total_m2/1e6:.3f} km^2")

    epochs_arr = np.array(EPOCHS)
    totals_arr = np.array(totals)
    slope, intercept, r_value, p_value, std_err = stats.linregress(epochs_arr, totals_arr)

    print(f"\n  Linear trend: {slope/1e3:+.2f} thousand m^2/year ({slope/1e6:+.4f} km^2/year)")
    print(f"  R-squared: {r_value**2:.3f}")
    print(f"  p-value: {p_value:.4f}  ({'SIGNIFICANT' if p_value < 0.05 else 'not significant'} at alpha=0.05)")
    total_change_pct = 100 * (totals_arr[-1] - totals_arr[0]) / totals_arr[0]
    print(f"  Total change {EPOCHS[0]}->{EPOCHS[-1]}: {total_change_pct:+.1f}%")

    # --- Step 2: resample first and last epoch onto 10m grid, compute growth mask ---
    ref_stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(ref_stack_path) as ref:
        ref_profile = ref.profile.copy()
        names = list(ref.descriptions)

    def load_and_resample(epoch):
        path = Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif"
        with rasterio.open(path) as src:
            src_arr = src.read(1).astype(np.float32)
            src_transform, src_crs = src.transform, src.crs
        dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        reproject(
            source=src_arr, destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
        return dst

    ghsl_first = load_and_resample(EPOCHS[0])
    ghsl_last = load_and_resample(EPOCHS[-1])

    # GHSL built_surface is m^2 of built-up area per 100m (1ha) cell -> threshold for "built-up pixel"
    # Using >10% of cell area (1000 m^2 out of 10000 m^2) as a reasonable "meaningfully built-up" cutoff
    BUILTUP_M2_THRESHOLD = 1000
    builtup_first = ghsl_first > BUILTUP_M2_THRESHOLD
    builtup_last = ghsl_last > BUILTUP_M2_THRESHOLD

    new_growth = builtup_last & (~builtup_first)  # pixels that became built-up between first and last epoch
    n_new_growth = new_growth.sum()
    print(f"\n  New built-up pixels ({EPOCHS[0]}->{EPOCHS[-1]}): {n_new_growth}")

    # --- Cross-reference with flood-prone mask ---
    with rasterio.open(Path("data/processed") / f"{region}_flood_prone_mask.tif") as src:
        flood_prone = src.read(1)

    growth_in_flood_prone = (new_growth & (flood_prone == 1)).sum()
    growth_in_safe = (new_growth & (flood_prone == 0)).sum()

    if n_new_growth > 0:
        pct_growth_flood_prone = 100 * growth_in_flood_prone / n_new_growth
        print(f"  Of new growth: {pct_growth_flood_prone:.1f}% occurred in flood-prone areas, "
              f"{100-pct_growth_flood_prone:.1f}% in flood-safe areas")

        # Compare to overall flood-prone share of land, to see if growth is DISPROPORTIONATE
        with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as _s:
            _names = list(_s.descriptions)
            _water = _s.read(_names.index("WATER_MASK") + 1) == 1
        overall_flood_prone_share = 100 * (flood_prone[~_water] == 1).mean()
        print(f"  (for reference: {overall_flood_prone_share:.1f}% of land, water excluded, in this region is flood-prone)")
        if pct_growth_flood_prone > overall_flood_prone_share:
            print(f"  -> Growth is DISPROPORTIONATELY concentrated in flood-prone terrain "
                  f"(+{pct_growth_flood_prone - overall_flood_prone_share:.1f} points vs baseline)")
        else:
            print(f"  -> Growth is NOT disproportionately in flood-prone terrain "
                  f"({pct_growth_flood_prone - overall_flood_prone_share:+.1f} points vs baseline)")
    else:
        print("  No new growth pixels detected between first and last epoch at this threshold.")
