"""
SHAP explainability for the Region-A-trained RF built-up classifier.
Analyzes:
  1. Global feature importance (mean |SHAP|) on Region A held-out test set
     and on Region B (cross-region transfer) test set -- side by side.
  2. Whether SHAP attributions differ between flood-exposed built-up
     pixels and non-exposed built-up pixels, in each region.
Uses a stratified random sample (not the full raster) for tractability.
"""
import rasterio
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

FEATURE_BANDS = [
    "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
    "NDVI", "NDBI", "S1_VV", "S1_VH", "DEM",
]
SAMPLE_SIZE = 4000
RNG = np.random.default_rng(42)

Path("results/figures").mkdir(parents=True, exist_ok=True)


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
    features = bands[feature_idx]
    return features, label, water, exposure


def stratified_sample(features, label, water, exposure, n_per_group):
    """Sample n_per_group pixels from each of: {non-built-up land, exposed built-up, non-exposed built-up}."""
    n_features, H, W = features.shape
    X_flat = features.reshape(n_features, -1).T
    y_flat = label.reshape(-1)
    water_flat = water.reshape(-1)
    exp_flat = exposure.reshape(-1)

    land = water_flat == 0
    non_builtup = land & (y_flat == 0)
    exposed_builtup = land & (y_flat == 1) & (exp_flat == 1)
    nonexposed_builtup = land & (y_flat == 1) & (exp_flat == 0)

    groups = {"non_builtup": non_builtup, "exposed_builtup": exposed_builtup, "nonexposed_builtup": nonexposed_builtup}
    X_parts, y_parts, group_labels = [], [], []
    for gname, gmask in groups.items():
        idx = np.where(gmask)[0]
        n = min(n_per_group, len(idx))
        chosen = RNG.choice(idx, size=n, replace=False)
        X_parts.append(X_flat[chosen])
        y_parts.append(y_flat[chosen])
        group_labels.extend([gname] * n)
        print(f"    sampled {n} from '{gname}' (available: {len(idx)})")

    return np.vstack(X_parts), np.concatenate(y_parts), np.array(group_labels)


print("Loading Random Forest (trained on Region A)...")
clf = joblib.load("models/rf_builtup_surrogate_shap.pkl")
explainer = shap.TreeExplainer(clf)

for region in ["region_a_kelani", "region_b_dakar"]:
    print(f"\n=== SHAP analysis: {region} ===")
    features, label, water, exposure = load_region_arrays(region)
    X_sample, y_sample, group_labels = stratified_sample(features, label, water, exposure, SAMPLE_SIZE)

    print("  Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    # For binary RF, shap_values is a list [class0, class1] or a 2D array depending on shap version
    if isinstance(shap_values, list):
        sv = shap_values[1]  # contribution toward "built-up" class
    else:
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

    # --- Global summary plot ---
    plt.figure()
    shap.summary_plot(sv, X_sample, feature_names=FEATURE_BANDS, show=False, plot_size=(8, 6))
    plt.title(f"{region} — SHAP summary (RF trained on Region A)")
    plt.tight_layout()
    out_path = f"results/figures/{region}_shap_summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # --- Mean |SHAP| by feature, split by exposed vs non-exposed built-up ---
    mean_abs_shap = np.abs(sv).mean(axis=0)
    print("  Global mean |SHAP| per feature:")
    for name, val in sorted(zip(FEATURE_BANDS, mean_abs_shap), key=lambda x: -x[1]):
        print(f"    {name:8s} {val:.4f}")

    exposed_mask = group_labels == "exposed_builtup"
    nonexposed_mask = group_labels == "nonexposed_builtup"

    if exposed_mask.sum() > 0 and nonexposed_mask.sum() > 0:
        mean_shap_exposed = np.abs(sv[exposed_mask]).mean(axis=0)
        mean_shap_nonexposed = np.abs(sv[nonexposed_mask]).mean(axis=0)

        print("\n  Feature importance: exposed built-up vs non-exposed built-up:")
        print(f"    {'feature':8s} {'exposed':>10s} {'non-exposed':>12s} {'diff':>8s}")
        for i, name in enumerate(FEATURE_BANDS):
            diff = mean_shap_exposed[i] - mean_shap_nonexposed[i]
            print(f"    {name:8s} {mean_shap_exposed[i]:>10.4f} {mean_shap_nonexposed[i]:>12.4f} {diff:>+8.4f}")

        # Bar chart comparing the two
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(FEATURE_BANDS))
        width = 0.35
        order = np.argsort(-mean_shap_exposed)
        ax.bar(x - width/2, mean_shap_exposed[order], width, label="Flood-exposed built-up")
        ax.bar(x + width/2, mean_shap_nonexposed[order], width, label="Non-exposed built-up")
        ax.set_xticks(x)
        ax.set_xticklabels(np.array(FEATURE_BANDS)[order], rotation=45, ha="right")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(f"{region} — feature importance: exposed vs non-exposed built-up")
        ax.legend()
        plt.tight_layout()
        out_path2 = f"results/figures/{region}_shap_exposed_vs_nonexposed.png"
        plt.savefig(out_path2, dpi=150)
        plt.close()
        print(f"  Saved: {out_path2}")
