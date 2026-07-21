"""
Computes HAND using a BUFFERED DEM (extra ~5.5km margin) so D8 flow
routing has real upstream/downstream context, avoiding the tile-boundary
NaN artifact seen when computing directly on the unbuffered crop. The
final HAND raster is clipped back down to the ORIGINAL region bounding
box before being resampled onto the 10m feature-stack grid.
"""
from pysheds.grid import Grid
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
import sys
sys.path.insert(0, "src")
from config.regions import REGIONS
from pathlib import Path

ACC_THRESHOLD = 500

for region, cfg in REGIONS.items():
    print(f"\n=== {region} ===")
    dem_path = Path("data/raw") / region / f"{region}_dem_buffered.tif"
    orig_bbox = cfg["bbox"]  # [minlon, minlat, maxlon, maxlat] -- ORIGINAL (unbuffered) AOI

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    print("  Filling pits and depressions...")
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)

    print("  Computing D8 flow direction...")
    fdir = grid.flowdir(inflated)

    print("  Computing flow accumulation...")
    acc = grid.accumulation(fdir)

    print(f"  Extracting stream network (accumulation > {ACC_THRESHOLD})...")
    streams = (acc > ACC_THRESHOLD)
    n_stream_cells = int(streams.sum())
    print(f"  Stream cells identified: {n_stream_cells} ({100*n_stream_cells/streams.size:.3f}% of buffered area)")

    print("  Computing HAND on buffered DEM...")
    hand = grid.compute_hand(fdir, inflated, streams)
    hand_array = np.asarray(hand, dtype=np.float32)
    affine = grid.affine
    crs = grid.crs

    n_nan_buffered = np.isnan(hand_array).sum()
    print(f"  NaN cells in buffered HAND: {n_nan_buffered} ({100*n_nan_buffered/hand_array.size:.2f}%)")

    # --- Write buffered HAND to a temp raster (needed so we can crop with rasterio) ---
    buffered_profile = {
        "driver": "GTiff", "height": hand_array.shape[0], "width": hand_array.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": affine, "nodata": -9999.0,
    }
    tmp_path = Path("data/processed") / f"{region}_hand_buffered_tmp.tif"
    hand_write = np.where(np.isnan(hand_array), -9999.0, hand_array)
    with rasterio.open(tmp_path, "w", **buffered_profile) as dst:
        dst.write(hand_write[np.newaxis, :, :])

    # --- Crop back down to the ORIGINAL (unbuffered) region bounding box ---
    minlon, minlat, maxlon, maxlat = orig_bbox
    with rasterio.open(tmp_path) as src:
        window = from_bounds(minlon, minlat, maxlon, maxlat, transform=src.transform)
        cropped = src.read(1, window=window)
        cropped_transform = src.window_transform(window)

    n_nan_cropped = (cropped == -9999.0).sum()
    print(f"  NaN cells after cropping to original AOI: {n_nan_cropped} ({100*n_nan_cropped/cropped.size:.2f}%)")

    native_profile = {
        "driver": "GTiff", "height": cropped.shape[0], "width": cropped.shape[1],
        "count": 1, "dtype": "float32", "crs": crs, "transform": cropped_transform,
        "nodata": -9999.0, "compress": "lzw",
    }
    out_path = Path("data/processed") / f"{region}_hand_native.tif"
    with rasterio.open(out_path, "w", **native_profile) as dst:
        dst.write(cropped[np.newaxis, :, :])
        dst.set_band_description(1, "HAND")
    print(f"  Saved cropped native-resolution HAND: {out_path}")

    # --- Resample onto the 10m feature-stack grid ---
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(stack_path) as ref:
        ref_profile = ref.profile.copy()

    hand_10m = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
    reproject(
        source=cropped,
        destination=hand_10m,
        src_transform=cropped_transform,
        src_crs=crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        src_nodata=-9999.0,
        dst_nodata=-9999.0,
        resampling=Resampling.bilinear,
    )

    out_10m_profile = ref_profile.copy()
    out_10m_profile.update(count=1, dtype="float32", compress="lzw", nodata=-9999.0)
    out_10m_path = Path("data/processed") / f"{region}_hand_10m.tif"
    with rasterio.open(out_10m_path, "w", **out_10m_profile) as dst:
        dst.write(hand_10m[np.newaxis, :, :])
        dst.set_band_description(1, "HAND")

    valid = hand_10m != -9999.0
    print(f"  Saved 10m-aligned HAND: {out_10m_path}")
    print(f"  HAND stats: min={hand_10m[valid].min():.2f} max={hand_10m[valid].max():.2f} mean={hand_10m[valid].mean():.2f}  remaining NaN%={100*(~valid).sum()/valid.size:.2f}")

    tmp_path.unlink()  # cleanup temp buffered file
