"""True-colour Sentinel-2 QA previews (Figures E.1/E.2) with scale bar, north arrow, CRS and source line."""
import math, textwrap
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REGIONS = {"region_a_kelani": "Region A - Lower Kelani Basin, Sri Lanka",
           "region_b_dakar": "Region B - Dakar northern suburbs, Senegal"}
Path("results/figures").mkdir(parents=True, exist_ok=True)
SOURCE = ("Contains Copernicus Sentinel-2 data (COPERNICUS/S2_SR_HARMONIZED), median composite 2024-01 to 2025-06, "
          "scene cloud cover < 20%; no pixel-level cloud mask applied.")


def stretch(band, low=2, high=98):
    lo, hi = np.percentile(band, [low, high])
    return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)


def furniture(ax, transform, crs, shape, source):
    h, w = shape
    if crs.is_geographic:
        lat = transform.f + transform.e * h / 2
        px_m = abs(transform.a) * 111320 * math.cos(math.radians(lat))
        approx = " (scale approximate)"
    else:
        px_m, approx = abs(transform.a), ""
    target_km = w * px_m / 1000 * 0.2
    nice = [0.5, 1, 2, 5, 10, 20, 50]
    km = max([n for n in nice if n <= target_km] or [nice[0]])
    bar_px = km * 1000 / px_m
    x0, y0 = 0.04 * w, 0.95 * h
    ax.add_patch(Rectangle((x0 - 0.015 * w, y0 - 0.055 * h), bar_px + 0.03 * w, 0.075 * h,
                           fc="white", ec="none", alpha=0.85, zorder=3))
    ax.plot([x0, x0 + bar_px], [y0, y0], color="k", lw=4, solid_capstyle="butt", zorder=4)
    ax.text(x0 + bar_px / 2, y0 - 0.012 * h, f"{km:g} km", ha="center", va="bottom", fontsize=11, zorder=5)
    ax.annotate("", xy=(0.93, 0.93), xytext=(0.93, 0.83), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"), zorder=5)
    ax.text(0.93, 0.945, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=14,
            fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    note = f"CRS: {crs.to_string() if crs else 'unknown'}{approx}. " + source
    ax.text(0, -0.015, textwrap.fill(note, 105), transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)


for region, label in REGIONS.items():
    path = Path("data/raw") / region / f"{region}_s2_composite.tif"
    with rasterio.open(path) as src:
        # Bands exported as [B2, B3, B4, B8] = Blue, Green, Red, NIR
        blue, green, red = (src.read(i).astype(float) for i in (1, 2, 3))
        tr, crs = src.transform, src.crs
    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb); ax.axis("off"); ax.set_autoscale_on(False)
    furniture(ax, tr, crs, rgb.shape[:2], SOURCE)
    ax.set_title(f"{label}\nSentinel-2 true colour (2024-01 to 2025-06 median)", fontsize=13)
    out = f"results/figures/{region}_s2_preview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("Saved preview:", out)
