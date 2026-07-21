import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
Path("results/figures").mkdir(parents=True, exist_ok=True)

for region in REGIONS:
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    exp_path = Path("data/processed") / f"{region}_exposure_map.tif"
    fp_path = Path("data/processed") / f"{region}_flood_prone_mask.tif"

    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        red = src.read(names.index("B4") + 1).astype(float)
        green = src.read(names.index("B3") + 1).astype(float)
        blue = src.read(names.index("B2") + 1).astype(float)

    with rasterio.open(exp_path) as src:
        exposure = src.read(1)
    with rasterio.open(fp_path) as src:
        flood_prone = src.read(1)

    def stretch(band, low=2, high=98):
        lo, hi = np.percentile(band, [low, high])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])

    overlay = rgb.copy()
    # Flood-prone (not built-up) -> blue tint
    fp_only = (flood_prone == 1) & (exposure == 0)
    overlay[fp_only] = overlay[fp_only] * 0.5 + np.array([0.2, 0.4, 1.0]) * 0.5
    # Exposed (flood-prone AND built-up) -> red tint
    overlay[exposure == 1] = overlay[exposure == 1] * 0.4 + np.array([1.0, 0.1, 0.1]) * 0.6

    plt.figure(figsize=(9, 9))
    plt.imshow(overlay)
    plt.title(f"{region} — flood exposure overlay\n(blue=flood-prone only, red=flood-prone + built-up)")
    plt.axis("off")
    out_path = f"results/figures/{region}_exposure_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
