import numpy as np, rasterio, csv
REGIONS = ["region_a_kelani", "region_b_dakar"]
THRESH = [1, 2, 3, 5, 7, 10]
def band_idx(desc, name, default):
    for i, d in enumerate(desc, 1):
        if d and name.lower() in d.lower():
            return i
    return default
rows = []
for r in REGIONS:
    with rasterio.open(f"data/processed/{r}_feature_stack.tif") as fs:
        print(r, "bands:", fs.count, fs.descriptions)
        hand = fs.read(band_idx(fs.descriptions, "HAND", 16)).astype("float32")
        water = fs.read(band_idx(fs.descriptions, "WATER", 15))
    with rasterio.open(f"data/processed/{r}_builtup_label.tif") as lb:
        built = lb.read(1) == 1
    assert hand.shape == built.shape == water.shape
    hand = np.nan_to_num(hand, nan=1e9)
    land = water == 0
    for t in THRESH:
        fp = (hand < t) & land
        exposed = fp & built
        rows.append([r, t, round(100*fp.sum()/land.sum(), 2), round(100*exposed.sum()/built.sum(), 2)])
with open("results/hand_threshold_sensitivity.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["region", "hand_threshold_m", "flood_prone_share_of_land_pct", "built_up_exposed_pct"])
    w.writerows(rows)
for row in rows: print(row)
print("CHECK: at 5 m expect ~39.1 and ~71.7 built-up exposed; ~43.5 and ~75.1 flood-prone land")
