"""
Render quick true-color PNG previews of the Sentinel-2 composites for
both regions, for a fast visual sanity check (cloud contamination,
obvious artifacts) that numeric CRS/bounds checks can't catch.
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
Path("results/figures").mkdir(parents=True, exist_ok=True)

for region in REGIONS:
    path = Path("data/raw") / region / f"{region}_s2_composite.tif"
    with rasterio.open(path) as src:
        # Bands were exported as [B2, B3, B4, B8] = Blue, Green, Red, NIR
        blue = src.read(1).astype(float)
        green = src.read(2).astype(float)
        red = src.read(3).astype(float)

    def stretch(band, low=2, high=98):
        lo, hi = np.percentile(band, [low, high])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(f"{region} — Sentinel-2 true color (2024-01 to 2025-06 median)")
    plt.axis("off")
    out_path = f"results/figures/{region}_s2_preview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved preview: {out_path}")
