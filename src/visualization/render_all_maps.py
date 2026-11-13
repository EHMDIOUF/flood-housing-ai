"""Re-renders the exposure, built-up and growth maps with scale bar, north arrow,
CRS note, readable legend and source line. Outputs overwrite the same PNG names."""
import math, textwrap
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

REGIONS = {"region_a_kelani": "Region A - Lower Kelani Basin, Sri Lanka",
           "region_b_dakar": "Region B - Dakar northern suburbs, Senegal"}
OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)

SRC_EXPOSURE = ("Contains Copernicus Sentinel-2 data (2024-2025); HAND derived from Copernicus GLO-30 DEM; "
                "built-up mask from a Random Forest trained on ESA WorldCover v200 (2021) and Open Buildings v3 (Google) labels.")
SRC_BUILTUP = ("Reference built-up labels: ESA WorldCover v200 (2021) and Open Buildings v3 (Google); "
               "water from the WorldCover water class.")
SRC_GROWTH = ("Built-up growth: JRC GHS-BUILT-S R2023A (100 m), 2000 and 2025; flood-prone mask: HAND from Copernicus GLO-30 DEM; "
              "background: Contains Copernicus Sentinel-2 data (2024-2025).")


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
        px_m = abs(transform.a)
        approx = ""
    target_km = w * px_m / 1000 * 0.2
    nice = [0.5, 1, 2, 5, 10, 20, 50]
    km = max([n for n in nice if n <= target_km] or [nice[0]])
    bar_px = km * 1000 / px_m
    x0, y0 = 0.04 * w, 0.95 * h
    ax.add_patch(Rectangle((x0 - 0.015 * w, y0 - 0.055 * h), bar_px + 0.03 * w, 0.075 * h,
                           fc="white", ec="none", alpha=0.85, zorder=3))
    ax.plot([x0, x0 + bar_px], [y0, y0], color="k", lw=4, solid_capstyle="butt", zorder=4)
    ax.text(x0 + bar_px / 2, y0 - 0.012 * h, f"{km:g} km", ha="center", va="bottom", fontsize=11, zorder=5)
    # north arrow (rasters are north-up)
    ax.annotate("", xy=(0.93, 0.93), xytext=(0.93, 0.83), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"), zorder=5)
    ax.text(0.93, 0.945, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=14,
            fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=5)
    crs_txt = crs.to_string() if crs else "unknown"
    note = f"CRS: {crs_txt}; 10 m pixel grid{approx}. " + source
    ax.text(0, -0.015, textwrap.fill(note, 115), transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)


def finish(fig, ax, title, handles, out_name):
    ax.set_title(title, fontsize=13)
    ax.legend(handles=handles, loc="lower right", framealpha=0.92, fontsize=11)
    fig.savefig(OUT / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT / out_name)


for region, label in REGIONS.items():
    with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as src:
        names = list(src.descriptions)
        red = src.read(names.index("B4") + 1).astype(float)
        green = src.read(names.index("B3") + 1).astype(float)
        blue = src.read(names.index("B2") + 1).astype(float)
        water = src.read(names.index("WATER_MASK") + 1)
        prof = src.profile.copy(); tr, crs = src.transform, src.crs
    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    shape = rgb.shape[:2]

    # ---- 1) exposure overlay
    with rasterio.open(Path("data/processed") / f"{region}_exposure_map.tif") as s: exposure = s.read(1)
    with rasterio.open(Path("data/processed") / f"{region}_flood_prone_mask.tif") as s: prone = s.read(1)
    ov = rgb.copy()
    fp_only = (prone == 1) & (exposure == 0)
    ov[fp_only] = ov[fp_only] * 0.5 + np.array([0.2, 0.4, 1.0]) * 0.5
    ov[exposure == 1] = ov[exposure == 1] * 0.4 + np.array([1.0, 0.1, 0.1]) * 0.6
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(ov); ax.axis("off"); ax.set_autoscale_on(False)
    furniture(ax, tr, crs, shape, SRC_EXPOSURE)
    finish(fig, ax, f"{label}\nFlood-prone terrain (HAND < 5 m) and exposed built-up area",
           [Patch(fc=(0.2, 0.4, 1.0), label="Flood-prone terrain (not built-up)"),
            Patch(fc=(1.0, 0.1, 0.1), label="Exposed: flood-prone and built-up")],
           f"{region}_exposure_overlay.png")

    # ---- 2) built-up classification map
    with rasterio.open(Path("data/processed") / f"{region}_builtup_label.tif") as s: builtup = s.read(1)
    img = np.zeros((*builtup.shape, 3), np.float32)
    img[water == 1] = [0.55, 0.75, 0.95]
    img[(water == 0) & (builtup == 0)] = [0.88, 0.93, 0.83]
    img[(water == 0) & (builtup == 1)] = [0.70, 0.15, 0.15]
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(img); ax.axis("off"); ax.set_autoscale_on(False)
    furniture(ax, tr, crs, shape, SRC_BUILTUP)
    finish(fig, ax, f"{label}\nBuilt-up / housing reference labels",
           [Patch(fc=(0.70, 0.15, 0.15), label="Built-up / housing"),
            Patch(fc=(0.88, 0.93, 0.83), label="Non-built-up land"),
            Patch(fc=(0.55, 0.75, 0.95), label="Water")],
           f"{region}_builtup_classification_map.png")

    # ---- 3) growth overlay
    def load(epoch):
        with rasterio.open(Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif") as s:
            arr, t, c = s.read(1).astype(np.float32), s.transform, s.crs
        dst = np.empty(shape, np.float32)
        reproject(source=arr, destination=dst, src_transform=t, src_crs=c,
                  dst_transform=prof["transform"], dst_crs=prof["crs"], resampling=Resampling.bilinear)
        return dst > 1000
    new = load(2025) & ~load(2000)
    ov = rgb.copy()
    safe, risk = new & (prone == 0), new & (prone == 1)
    ov[safe] = ov[safe] * 0.3 + np.array([1.0, 0.85, 0.0]) * 0.7
    ov[risk] = ov[risk] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(ov); ax.axis("off"); ax.set_autoscale_on(False)
    furniture(ax, tr, crs, shape, SRC_GROWTH)
    finish(fig, ax, f"{label}\nNew built-up growth 2000-2025 (GHSL)",
           [Patch(fc=(1.0, 0.85, 0.0), label="New growth in flood-safe area"),
            Patch(fc=(1.0, 0.0, 0.0), label="New growth in flood-prone area")],
           f"{region}_growth_flood_overlay.png")
