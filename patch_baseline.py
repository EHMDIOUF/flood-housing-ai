from pathlib import Path
import shutil

def patch(path, pairs):
    p = Path(path)
    s = p.read_text()
    shutil.copy(p, str(p) + ".bak")
    for old, new in pairs:
        assert s.count(old) == 1, f"{path}: expected exactly 1 match for:\n{old}"
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", path)

# 1) script: land-only baseline (water excluded)
patch("src/trend_modelling/analyze_ghsl_trend.py", [(
'''        overall_flood_prone_share = 100 * flood_prone.sum() / flood_prone.size
        print(f"  (for reference: {overall_flood_prone_share:.1f}% of all land in this region is flood-prone)")''',
'''        with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as _s:
            _names = list(_s.descriptions)
            _water = _s.read(_names.index("WATER_MASK") + 1) == 1
        overall_flood_prone_share = 100 * (flood_prone[~_water] == 1).mean()
        print(f"  (for reference: {overall_flood_prone_share:.1f}% of land, water excluded, in this region is flood-prone)")''')])

# 2) dashboard
patch("src/dashboard/app.py", [
# Time Comparison page: grouped bar with land-share baseline
('''            fig = go.Figure(go.Bar(x=[SHORT[r] for r in RK], y=shares, marker_color=[T["bad"]] * 2, text=[f"{v:.1f}%" for v in shares], textposition="outside"))
            fig.update_layout(yaxis_range=[0, max(shares) * 1.3 + 1])''',
'''            base = [43.5, 75.1]  # flood-prone share of land (water excluded), Region A, Region B
            assert len(RK) == 2
            fig = go.Figure([
                go.Bar(name="Share of new growth in flood-prone terrain", x=[SHORT[r] for r in RK], y=shares, marker_color=[T["bad"]] * 2, text=[f"{v:.1f}%" for v in shares], textposition="outside"),
                go.Bar(name="Flood-prone share of land (water excluded)", x=[SHORT[r] for r in RK], y=base, marker_color="#94a3b8", text=[f"{v:.1f}%" for v in base], textposition="outside"),
            ])
            fig.update_layout(barmode="group", yaxis_range=[0, max(shares + base) * 1.3 + 1], legend=dict(orientation="h", y=-0.25))'''),
# Growth page KPI: show the baseline in the sub-label
('''("Growth inside flood-prone zones", f"{G['prone_share']:.1f}%", "of new built-up area", "dn")''',
'''("Growth inside flood-prone zones", f"{G['prone_share']:.1f}%", f"of new built-up area; land share {dict(zip(RK, (43.5, 75.1)))[region]:.1f}%", "dn")'''),
])
