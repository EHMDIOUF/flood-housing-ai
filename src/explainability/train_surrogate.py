"""
Trains a shallow surrogate Random Forest (same features, same training
data as the production model) specifically for SHAP explainability.
The production model (max_depth=20, ~43k leaves/tree) is too large for
tractable exact TreeSHAP. This surrogate uses max_depth=8, which should
have orders of magnitude fewer leaves per tree while still capturing
most of the predictive signal.

Reports: surrogate's own accuracy on the same held-out test sets used
for the production model, AND agreement rate with the production
model's predictions (how often they agree pixel-for-pixel) -- both
matter for trusting the surrogate's explanations as representative.
"""
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from pathlib import Path
import joblib

FEATURE_BANDS = [
    "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
    "NDVI", "NDBI", "S1_VV", "S1_VH", "DEM",
]
TILE_SIZE = 64


def load_region(region):
    stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
    label_path = Path("data/processed") / f"{region}_builtup_label.tif"
    with rasterio.open(stack_path) as src:
        names = list(src.descriptions)
        bands = src.read()
        water_mask = bands[names.index("WATER_MASK")]
    with rasterio.open(label_path) as src:
        label = src.read(1)
    feature_idx = [names.index(b) for b in FEATURE_BANDS]
    return bands[feature_idx], label, water_mask


def checkerboard_mask(H, W, tile_size):
    row_tile = (np.arange(H) // tile_size)[:, None]
    col_tile = (np.arange(W) // tile_size)[None, :]
    return ((row_tile + col_tile) % 2 == 0)


def to_pixel_table(features, label, water_mask, mask_extra=None):
    n_features, H, W = features.shape
    X = features.reshape(n_features, -1).T
    y = label.reshape(-1)
    water = water_mask.reshape(-1)
    valid = water == 0
    if mask_extra is not None:
        valid = valid & mask_extra.reshape(-1)
    return X[valid], y[valid]


print("Loading Region A (training)...")
feat_a, label_a, water_a = load_region("region_a_kelani")
H, W = label_a.shape
train_mask = checkerboard_mask(H, W, TILE_SIZE)
test_mask = ~train_mask

X_train, y_train = to_pixel_table(feat_a, label_a, water_a, mask_extra=train_mask)
X_test_a, y_test_a = to_pixel_table(feat_a, label_a, water_a, mask_extra=test_mask)

print("Training shallow surrogate Random Forest (max_depth=8)...")
surrogate = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=20,
    n_jobs=-1, random_state=42, class_weight="balanced",
)
surrogate.fit(X_train, y_train)

leaf_counts = [tree.tree_.n_leaves for tree in surrogate.estimators_]
print(f"Surrogate leaf count per tree: min={min(leaf_counts)} max={max(leaf_counts)} mean={sum(leaf_counts)/len(leaf_counts):.0f}")

# --- Surrogate's own accuracy (same metrics as production model eval) ---
pred_a = surrogate.predict(X_test_a)
print(f"\nSurrogate accuracy on Region A held-out:")
print(f"  Balanced accuracy: {balanced_accuracy_score(y_test_a, pred_a):.4f}")
print(f"  Macro F1:          {f1_score(y_test_a, pred_a, average='macro'):.4f}")

print("\nLoading Region B (transfer test)...")
feat_b, label_b, water_b = load_region("region_b_dakar")
X_test_b, y_test_b = to_pixel_table(feat_b, label_b, water_b)
pred_b = surrogate.predict(X_test_b)
print(f"Surrogate accuracy on Region B (transfer):")
print(f"  Balanced accuracy: {balanced_accuracy_score(y_test_b, pred_b):.4f}")
print(f"  Macro F1:          {f1_score(y_test_b, pred_b, average='macro'):.4f}")

# --- Agreement with production model ---
production = joblib.load("models/rf_builtup_trainA.pkl")
prod_pred_a = production.predict(X_test_a)
prod_pred_b = production.predict(X_test_b)

agreement_a = accuracy_score(prod_pred_a, pred_a)
agreement_b = accuracy_score(prod_pred_b, pred_b)
print(f"\nSurrogate vs production agreement rate:")
print(f"  Region A: {agreement_a:.4f}")
print(f"  Region B: {agreement_b:.4f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(surrogate, "models/rf_builtup_surrogate_shap.pkl")
print("\nSurrogate saved to models/rf_builtup_surrogate_shap.pkl")
