"""
Bootstrap significance test: for each feature, is the mean |SHAP|
difference between flood-exposed and non-exposed built-up pixels
larger than we'd expect from sampling noise alone?

Resamples both groups with replacement many times, recomputes the mean
difference each time, and reports a 95% confidence interval. If the CI
excludes zero, the difference is likely real; if it includes zero,
it's consistent with noise.
"""
import rasterio
import numpy as np
import joblib
import shap
from pathlib import Path

FEATURE_BANDS = [
    "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
    "NDVI", "NDBI", "S1_VV", "S1_VH", "DEM",
]
SAMPLE_SIZE = 4000
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def load_region_arrays(region):
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    label_path = Path("data/processed") / f"{region}_builtup_label.tif"
    exposure_path = Path("data/processed") / f"{region}_exposure_map.tif"
    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        bands = src.read()
        water = bands[names.index("WATER_MASK")]
    with rasterio.open(label_path) as src:
        label = src.read(1)
    with rasterio.open(exposure_path) as src:
        exposure = src.read(1)
    feature_idx = [names.index(b) for b in FEATURE_BANDS]
    return bands[feature_idx], label, water, exposure


def sample_group(features, label, water, exposure, group, n):
    n_features, H, W = features.shape
    X_flat = features.reshape(n_features, -1).T
    y_flat = label.reshape(-1)
    water_flat = water.reshape(-1)
    exp_flat = exposure.reshape(-1)
    land = water_flat == 0
    if group == "exposed":
        mask = land & (y_flat == 1) & (exp_flat == 1)
    else:
        mask = land & (y_flat == 1) & (exp_flat == 0)
    idx = np.where(mask)[0]
    chosen = RNG.choice(idx, size=min(n, len(idx)), replace=False)
    return X_flat[chosen]


print("Loading surrogate model...")
clf = joblib.load("models/rf_builtup_surrogate_shap.pkl")
explainer = shap.TreeExplainer(clf)

for region in ["region_a_kelani", "region_b_dakar"]:
    print(f"\n=== {region} ===")
    features, label, water, exposure = load_region_arrays(region)

    X_exposed = sample_group(features, label, water, exposure, "exposed", SAMPLE_SIZE)
    X_nonexposed = sample_group(features, label, water, exposure, "nonexposed", SAMPLE_SIZE)

    print("  Computing SHAP values for both groups...")
    sv_exposed = explainer.shap_values(X_exposed)
    sv_nonexposed = explainer.shap_values(X_nonexposed)
    sv_exposed = sv_exposed[1] if isinstance(sv_exposed, list) else (sv_exposed[:, :, 1] if sv_exposed.ndim == 3 else sv_exposed)
    sv_nonexposed = sv_nonexposed[1] if isinstance(sv_nonexposed, list) else (sv_nonexposed[:, :, 1] if sv_nonexposed.ndim == 3 else sv_nonexposed)

    abs_exposed = np.abs(sv_exposed)
    abs_nonexposed = np.abs(sv_nonexposed)

    print(f"  Bootstrapping ({N_BOOTSTRAP} resamples)...")
    n_e, n_ne = len(abs_exposed), len(abs_nonexposed)
    boot_diffs = np.zeros((N_BOOTSTRAP, len(FEATURE_BANDS)))
    for b in range(N_BOOTSTRAP):
        idx_e = RNG.integers(0, n_e, n_e)
        idx_ne = RNG.integers(0, n_ne, n_ne)
        boot_diffs[b] = abs_exposed[idx_e].mean(axis=0) - abs_nonexposed[idx_ne].mean(axis=0)

    observed_diff = abs_exposed.mean(axis=0) - abs_nonexposed.mean(axis=0)
    ci_low = np.percentile(boot_diffs, 2.5, axis=0)
    ci_high = np.percentile(boot_diffs, 97.5, axis=0)

    print(f"\n  {'feature':8s} {'diff':>8s} {'95% CI low':>12s} {'95% CI high':>12s}  significant?")
    for i, name in enumerate(FEATURE_BANDS):
        sig = "YES" if (ci_low[i] > 0 or ci_high[i] < 0) else "no"
        print(f"  {name:8s} {observed_diff[i]:>+8.4f} {ci_low[i]:>+12.4f} {ci_high[i]:>+12.4f}  {sig}")
