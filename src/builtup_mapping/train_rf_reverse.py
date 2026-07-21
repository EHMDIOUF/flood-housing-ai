"""
Reverse-direction transferability check: trains on Region B (Dakar) using
a checkerboard spatial split, evaluates on B's own held-out tiles
(same-region baseline) and on full Region A (Kelani) as the transfer test.
Mirrors train_rf.py exactly, just with A and B swapped, so results are
directly comparable.
"""
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, f1_score, confusion_matrix,
)
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


def evaluate(name, clf, X, y):
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    bal_acc = balanced_accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, pred)
    print(f"\n--- {name} ---")
    print(f"  n={len(y)}  built-up rate={y.mean():.4f}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}   <-- fair metric across class balance")
    print(f"  Macro F1:          {macro_f1:.4f}   <-- fair metric across class balance")
    print(f"  Precision (pos):   {prec:.4f}")
    print(f"  Recall (pos):      {rec:.4f}")
    print(f"  F1 (pos class):    {f1:.4f}")
    print(f"  Confusion matrix [[TN, FP],[FN, TP]]:\n{cm}")


print("Loading Region B (training)...")
feat_b, label_b, water_b = load_region("region_b_dakar")
H, W = label_b.shape
train_mask = checkerboard_mask(H, W, TILE_SIZE)
test_mask = ~train_mask

X_train, y_train = to_pixel_table(feat_b, label_b, water_b, mask_extra=train_mask)
X_test_b, y_test_b = to_pixel_table(feat_b, label_b, water_b, mask_extra=test_mask)

print(f"Train pixels: {len(y_train)}  (built-up rate: {y_train.mean():.4f})")
print(f"Held-out Region B test pixels: {len(y_test_b)}  (built-up rate: {y_test_b.mean():.4f})")

print("\nTraining Random Forest...")
clf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=42, class_weight="balanced",
)
clf.fit(X_train, y_train)

evaluate("Region B held-out checkerboard tiles (same-region baseline)", clf, X_test_b, y_test_b)

print("\nLoading Region A (cross-region transfer test)...")
feat_a, label_a, water_a = load_region("region_a_kelani")
X_test_a, y_test_a = to_pixel_table(feat_a, label_a, water_a)
evaluate("Region A full (cross-region transfer)", clf, X_test_a, y_test_a)

Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/rf_builtup_trainB.pkl")

importances = clf.feature_importances_
print("\n--- Feature importances ---")
for name, imp in sorted(zip(FEATURE_BANDS, importances), key=lambda x: -x[1]):
    print(f"  {name:8s} {imp:.4f}")

print("\nModel saved to models/rf_builtup_trainB.pkl")
