"""Recomputes the encroachment baseline three ways (all pixels, land only, 2000 built-up)."""
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from pathlib import Path

for region in ["region_a_kelani", "region_b_dakar"]:
    with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as s:
        names = list(s.descriptions)
        water = s.read(names.index("WATER_MASK") + 1) == 1
        prof = s.profile
    with rasterio.open(Path("data/processed") / f"{region}_flood_prone_mask.tif") as s:
        prone = s.read(1) == 1

    def load(epoch):
        with rasterio.open(Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif") as s:
            arr, t, c = s.read(1).astype(np.float32), s.transform, s.crs
        dst = np.empty((prof["height"], prof["width"]), np.float32)
        reproject(arr, dst, src_transform=t, src_crs=c, dst_transform=prof["transform"],
                  dst_crs=prof["crs"], resampling=Resampling.bilinear)
        return dst > 1000

    g0, g1 = load(2000), load(2025)
    new = g1 & ~g0
    land = ~water
    print(f"\n=== {region} ===")
    print(f"  Share of new growth in flood-prone land:       {100*(new & prone).sum()/new.sum():.1f}%")
    print(f"  Baseline A: flood-prone share of ALL pixels:   {100*prone.mean():.1f}%   (current thesis baseline)")
    print(f"  Baseline B: flood-prone share of LAND only:    {100*prone[land].mean():.1f}%")
    print(f"  Baseline C: flood-prone share of 2000 built-up: {100*(g0 & prone).sum()/g0.sum():.1f}%")
    print(f"  Water share of the study box:                  {100*water.mean():.1f}%")
