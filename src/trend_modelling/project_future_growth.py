# --- Phase 4 / Step 6: illustrative linear extrapolation of GHSL trend to future years ---
import numpy as np
from scipy import stats
import json
from pathlib import Path

with open(Path("data/processed") / "dashboard_kpis.json") as f:
    kpis = json.load(f)

TARGET_YEARS = [2030, 2035]
REGIONS = {"region_a_kelani": "Region A - Kelani", "region_b_dakar": "Region B - Dakar"}

projection_results = {}

for region_key, label in REGIONS.items():
    k = kpis[region_key]
    years = np.array(k["ghsl_epochs"], dtype=float)
    km2 = np.array(k["ghsl_builtup_km2_by_epoch"], dtype=float)
    exposure_ratio = k["pct_builtup_exposed"] / 100.0

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, km2)
    n = len(years); x_mean = years.mean()
    ss_x = np.sum((years - x_mean) ** 2)
    mse = np.sum((km2 - (slope * years + intercept)) ** 2) / (n - 2)

    region_proj = {}
    for ty in TARGET_YEARS:
        point_est = slope * ty + intercept
        se_pred = np.sqrt(mse * (1 + 1/n + (ty - x_mean) ** 2 / ss_x))
        t_crit = stats.t.ppf(0.975, df=n - 2)
        margin = t_crit * se_pred
        low, high = point_est - margin, point_est + margin
        region_proj[ty] = {
            "builtup_km2": round(point_est, 1),
            "builtup_km2_95pi": [round(low, 1), round(high, 1)],
            "exposed_builtup_km2": round(point_est * exposure_ratio, 1),
            "exposed_builtup_km2_95pi": [round(low * exposure_ratio, 1), round(high * exposure_ratio, 1)],
        }
        print(f"{label} {ty}: {point_est:.1f} km2 built-up "
              f"(95% PI {low:.1f}-{high:.1f}), "
              f"~{point_est*exposure_ratio:.1f} km2 exposed at constant "
              f"{exposure_ratio*100:.1f}% ratio")

    projection_results[region_key] = region_proj

with open(Path("data/processed") / "growth_projection_2030_2035.json", "w") as f:
    json.dump(projection_results, f, indent=2)
print("\nSaved data/processed/growth_projection_2030_2035.json")
