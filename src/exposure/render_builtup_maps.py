"""
Renders clean built-up classification maps for Figures 4.2/4.3:
built-up = solid color, non-built-up land = light background, water = blue.
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]
Path("results/figures").mkdir(parents=True, exist_ok=True)

for region in REGIONS:
    with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as src:
        names = list(src.descriptions)
        water = src.read(names.index("WATER_MASK") + 1)

    with rasterio.open(Path("data/processed") / f"{region}_builtup_label.tif") as src:
        builtup = src.read(1)

    # Build an RGB classification image: water=blue, built-up=dark red, non-built-up land=light green
    img = np.zeros((*builtup.shape, 3), dtype=np.float32)
    img[water == 1] = [0.55, 0.75, 0.95]           # water - light blue
    land_nonbuiltup = (water == 0) & (builtup == 0)
    img[land_nonbuiltup] = [0.88, 0.93, 0.83]       # non-built-up land - pale green
    land_builtup = (water == 0) & (builtup == 1)
    img[land_builtup] = [0.70, 0.15, 0.15]          # built-up - dark red

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img)
    ax.axis("off")
    region_label = "Region A - Lower Kelani Basin" if region == "region_a_kelani" else "Region B - Dakar Suburbs"
    ax.set_title(f"{region_label}: Built-up / Housing Classification", fontsize=13)

    # Simple legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.70, 0.15, 0.15], label="Built-up / housing"),
        Patch(facecolor=[0.88, 0.93, 0.83], label="Non-built-up land"),
        Patch(facecolor=[0.55, 0.75, 0.95], label="Water"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=10)

    out_path = f"results/figures/{region}_builtup_classification_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
