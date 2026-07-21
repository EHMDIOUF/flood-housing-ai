import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
Path("results/figures").mkdir(parents=True, exist_ok=True)

for region in REGIONS:
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        red = src.read(names.index("B4") + 1).astype(float)
        green = src.read(names.index("B3") + 1).astype(float)
        blue = src.read(names.index("B2") + 1).astype(float)
        ref_profile = src.profile.copy()

    def load_and_resample(epoch):
        path = Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif"
        with rasterio.open(path) as s:
            arr = s.read(1).astype(np.float32)
            t, c = s.transform, s.crs
        dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        reproject(source=arr, destination=dst, src_transform=t, src_crs=c,
                   dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                   resampling=Resampling.bilinear)
        return dst

    ghsl_2000 = load_and_resample(2000)
    ghsl_2025 = load_and_resample(2025)
    builtup_2000 = ghsl_2000 > 1000
    builtup_2025 = ghsl_2025 > 1000
    new_growth = builtup_2025 & (~builtup_2000)

    with rasterio.open(Path("data/processed") / f"{region}_flood_prone_mask.tif") as src:
        flood_prone = src.read(1)

    def stretch(band, low=2, high=98):
        lo, hi = np.percentile(band, [low, high])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    overlay = rgb.copy()

    growth_safe = new_growth & (flood_prone == 0)
    growth_flood_prone = new_growth & (flood_prone == 1)

    overlay[growth_safe] = overlay[growth_safe] * 0.3 + np.array([1.0, 0.85, 0.0]) * 0.7   # yellow: growth, safe
    overlay[growth_flood_prone] = overlay[growth_flood_prone] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7  # red: growth, flood-prone

    plt.figure(figsize=(9, 9))
    plt.imshow(overlay)
    plt.title(f"{region} — new built-up growth 2000-2025\n(yellow=growth in safe area, red=growth in flood-prone area)")
    plt.axis("off")
    out_path = f"results/figures/{region}_growth_flood_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
