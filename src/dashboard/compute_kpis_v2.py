"""
Extends dashboard_kpis.json with:
  - Risk tiers (High/Medium/Low) based on HAND thresholds, as % of built-up area
  - Estimated population exposed (WorldPop summed over exposed pixels)
  - Per-epoch GHSL built-up delta (for year-slider change visualization)
"""
import rasterio
import numpy as np
import json
from pathlib import Path

REGIONS = ["region_a_kelani", "region_b_dakar"]

# Risk tiers based on HAND (m): lower = higher risk
HIGH_RISK_MAX = 2.0
MEDIUM_RISK_MAX = 5.0

with open(Path("data/processed") / "dashboard_kpis.json") as f:
    kpis = json.load(f)

for region in REGIONS:
    print(f"\n=== {region} ===")
    with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as src:
        names = list(src.descriptions)
        hand = src.read(names.index("HAND") + 1)
        worldpop = src.read(names.index("WorldPop") + 1)
        water = src.read(names.index("WATER_MASK") + 1)

    with rasterio.open(Path("data/processed") / f"{region}_builtup_label.tif") as src:
        builtup = src.read(1)
    with rasterio.open(Path("data/processed") / f"{region}_exposure_map.tif") as src:
        exposure = src.read(1)

    land = water == 0
    builtup_land = (builtup == 1) & land
    n_builtup = builtup_land.sum()

    high_risk = builtup_land & (hand < HIGH_RISK_MAX)
    medium_risk = builtup_land & (hand >= HIGH_RISK_MAX) & (hand < MEDIUM_RISK_MAX)
    low_risk = builtup_land & (hand >= MEDIUM_RISK_MAX)

    pct_high = 100 * high_risk.sum() / n_builtup
    pct_medium = 100 * medium_risk.sum() / n_builtup
    pct_low = 100 * low_risk.sum() / n_builtup

    print(f"  Risk tiers (of built-up area): High={pct_high:.1f}% Medium={pct_medium:.1f}% Low={pct_low:.1f}%")

    pop_exposed = float(worldpop[exposure == 1].sum())
    pop_total_builtup = float(worldpop[builtup_land].sum())
    print(f"  Estimated population exposed: {pop_exposed:,.0f} (of ~{pop_total_builtup:,.0f} in built-up areas)")

    kpis[region]["risk_pct_high"] = float(pct_high)
    kpis[region]["risk_pct_medium"] = float(pct_medium)
    kpis[region]["risk_pct_low"] = float(pct_low)
    kpis[region]["population_exposed"] = pop_exposed
    kpis[region]["population_total_builtup"] = pop_total_builtup

with open(Path("data/processed") / "dashboard_kpis.json", "w") as f:
    json.dump(kpis, f, indent=2)

print("\nSaved updated dashboard_kpis.json")
