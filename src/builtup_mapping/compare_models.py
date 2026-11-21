"""
compare_models.py  -  facts for the thesis corrections
Run from the repo root:   python3 src/builtup_mapping/compare_models.py
(copy this file to src/builtup_mapping/ first)

Produces
  results/pixel_counts.csv      pixels per region (valid, train, test, built-up rate, excluded)
  results/model_comparison.csv  same split / same metrics for several classifiers,
                                same-region (held-out tiles) and cross-region (other region)
Uses exactly the same inputs, checkerboard split (64 x 64 tiles) and water exclusion as train_rf.py.
Slow models (SVM, kNN) are trained on a random subsample (see SUBSAMPLE_*) and every model is
scored on the SAME random evaluation subsample, so the comparison is fair and finishes in minutes.
"""
import time, csv
from pathlib import Path
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             jaccard_score, confusion_matrix)

TILE = 64
RNG = np.random.default_rng(42)
SUBSAMPLE_TRAIN_SLOW = 60_000      # SVM / kNN / MLP training rows
SUBSAMPLE_TRAIN_FAST = 600_000     # logistic regression / boosting / RF
EVAL_ROWS = 300_000                # same evaluation rows for every model
REGIONS = ["region_a_kelani", "region_b_dakar"]
NON_PREDICTORS = {"worldcover", "worldpop", "water_mask", "hand"}   # in the 16-band stack, not classifier inputs
P = Path("data/processed"); OUT = Path("results"); OUT.mkdir(exist_ok=True)


def load(region):
    with rasterio.open(P / f"{region}_feature_stack.tif") as s:
        names = list(s.descriptions); arr = s.read().astype(np.float32)
    with rasterio.open(P / f"{region}_builtup_label.tif") as s:
        y = s.read(1)
    feat_idx = [i for i, n in enumerate(names) if (n or "").lower() not in NON_PREDICTORS]
    water = arr[[i for i, n in enumerate(names) if (n or "").lower() == "water_mask"][0]] == 1
    X = arr[feat_idx]
    assert len(feat_idx) == 12, f"expected 12 predictors, got {len(feat_idx)}: {[names[i] for i in feat_idx]}"
    H, W = y.shape
    rows, cols = np.indices((H, W))
    train_tile = ((rows // TILE) + (cols // TILE)) % 2 == 0
    finite = np.isfinite(X).all(axis=0) & np.isfinite(y)
    valid = finite & ~water
    return X, y, valid, train_tile, [names[i] for i in feat_idx], int((~finite).sum()), int(water.sum())


def table(X, y, mask):
    return X[:, mask].T, (y[mask] > 0).astype(int)


def metrics(y, p):
    cm = confusion_matrix(y, p, labels=[0, 1])
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro"), f1_builtup=f1_score(y, p),
                iou_builtup=jaccard_score(y, p), tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1])


def sub(X, y, n):
    if len(y) <= n: return X, y
    i = RNG.choice(len(y), n, replace=False); return X[i], y[i]


MODELS = {
    "Random Forest (thesis model)": (lambda: RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight="balanced"), "fast"),
    "Gradient boosting (HistGB)":   (lambda: HistGradientBoostingClassifier(max_iter=200, random_state=42, class_weight="balanced"), "fast"),
    "Logistic regression":          (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1)), "fast"),
    "SVM (RBF kernel)":             (lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")), "slow"),
    "k-Nearest Neighbours (k=5)":   (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)), "slow"),
    "Neural network (MLP, pixel-wise)": (lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=42)), "slow"),
}

data = {r: load(r) for r in REGIONS}

# ---- 1) pixel counts ------------------------------------------------------------------
with open(OUT / "pixel_counts.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["region", "total_pixels", "valid_pixels", "train_pixels", "test_pixels",
                                   "builtup_rate_train", "builtup_rate_test", "nodata_pixels_excluded", "water_pixels"])
    for r in REGIONS:
        X, y, valid, tr, names, nodata, water = data[r]
        ytr = (y[valid & tr] > 0); yte = (y[valid & ~tr] > 0)
        row = [r, y.size, int(valid.sum()), int((valid & tr).sum()), int((valid & ~tr).sum()),
               round(ytr.mean(), 4), round(yte.mean(), 4), nodata, water]
        w.writerow(row); print(row)
    print("Predictors:", data[REGIONS[0]][4])

# ---- 2) model comparison --------------------------------------------------------------
rows = []
for src in REGIONS:
    Xs, ys, vs, trs = data[src][:4]
    tgt = [r for r in REGIONS if r != src][0]
    Xt, yt, vt, _ = data[tgt][:4]
    Xtr, ytr = table(Xs, ys, vs & trs)
    Xte_same, yte_same = table(Xs, ys, vs & ~trs)
    Xte_x, yte_x = table(Xt, yt, vt)
    Xte_same, yte_same = sub(Xte_same, yte_same, EVAL_ROWS)
    Xte_x, yte_x = sub(Xte_x, yte_x, EVAL_ROWS)
    for name, (make, speed) in MODELS.items():
        n = SUBSAMPLE_TRAIN_SLOW if speed == "slow" else SUBSAMPLE_TRAIN_FAST
        Xa, ya = sub(Xtr, ytr, n)
        m = make(); t0 = time.time(); m.fit(Xa, ya); fit_s = time.time() - t0
        for setting, (Xe, ye) in {"same-region (held-out tiles)": (Xte_same, yte_same),
                                  f"cross-region (to {tgt})": (Xte_x, yte_x)}.items():
            t0 = time.time(); p = m.predict(Xe); pred_s = time.time() - t0
            r = dict(trained_on=src, model=name, setting=setting, train_rows=len(ya), eval_rows=len(ye),
                     fit_seconds=round(fit_s, 1), predict_seconds=round(pred_s, 1)); r.update(metrics(ye, p)); rows.append(r)
            print(f"{src:16s} {name:34s} {setting:34s} bal.acc={r['balanced_accuracy']:.3f} macroF1={r['macro_f1']:.3f} fit={fit_s:.0f}s")

with open(OUT / "model_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nSaved results/pixel_counts.csv and results/model_comparison.csv")

