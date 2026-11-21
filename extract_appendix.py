"""Collects facts for thesis Appendices A and B from the REAL project files.
Run from the project root. Writes appendix_material.md. Nothing is invented:
if something is not found it is reported as NOT FOUND."""
import ast, csv, json, platform, re, subprocess, sys
from importlib import metadata
from pathlib import Path

ROOT = Path.cwd()
OUT = []
IGN = {"data", ".git", "__pycache__", "venv", ".venv", "node_modules", ".idea", ".vscode"}
w = lambda s="": OUT.append(s)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"(failed: {e})"


def md_table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    w()


# ---------------------------------------------------------------- PART 1.1 tree
w("# APPENDIX A MATERIAL (generated from the real project)\n")
w("## A.1 Project folder structure (depth 3, data/.git/venv/__pycache__ ignored)\n")
w("```")
def walk(d, prefix="", depth=0):
    if depth >= 3:
        return
    items = sorted([p for p in d.iterdir() if p.name not in IGN and not p.name.endswith(".pyc")],
                   key=lambda p: (p.is_file(), p.name))
    for i, p in enumerate(items):
        last = i == len(items) - 1
        w(f"{prefix}{'└── ' if last else '├── '}{p.name}{'/' if p.is_dir() else ''}")
        if p.is_dir():
            walk(p, prefix + ("    " if last else "│   "), depth + 1)
w(ROOT.name + "/")
walk(ROOT)
w("```\n")

# ---------------------------------------------------------------- PART 1.2 script table
STAGE = {"ingestion": "S1 data ingestion (Earth Engine)", "preprocessing": "S2 preprocessing / feature stack",
         "builtup_mapping": "S3 Random Forest built-up classification", "flood_layer": "S4 HAND flood-prone layer",
         "exposure": "S5 exposure analysis", "explainability": "S6 SHAP explainability",
         "trend_modelling": "S7 GHSL growth trend and projection", "dashboard": "Streamlit dashboard"}
PATH_RE = re.compile(r"""["']([^"'\n]*(?:data/|results/|outputs/|models/)[^"'\n]*|[^"'\n]*\.(?:tif|json|csv|png|pkl|joblib|geojson|shp|npy|npz)[^"'\n]*)["']""")
OUT_WORDS = re.compile(r"savefig|to_csv|json\.dump|\.dump\(|\"w\"|'w'|write|imsave|Export\.|to_drive|to_asset|\.save\(|joblib\.dump")
w("## A.2 Script inventory\n")
w("Stage comes from the folder name. Inputs/outputs are found by scanning each file for path strings "
  "(heuristic: a path on a line that writes/saves/exports is listed as output). Check each row once.\n")
rows = []
pyfiles = sorted(p for p in ROOT.rglob("*.py") if not (set(p.relative_to(ROOT).parts) & IGN))
for p in pyfiles:
    txt = p.read_text(errors="ignore")
    try:
        doc = (ast.get_docstring(ast.parse(txt)) or "").strip().split("\n")[0]
    except SyntaxError:
        doc = ""
    if not doc:
        m = re.search(r"^#\s*(.+)$", txt, re.M)
        doc = m.group(1) if m else "NO DOCSTRING OR COMMENT FOUND"
    ins, outs = [], []
    for line in txt.splitlines():
        for s in PATH_RE.findall(line):
            (outs if OUT_WORDS.search(line) else ins).append(s.strip())
    ins = sorted(set(ins) - set(outs))[:6]
    outs = sorted(set(outs))[:6]
    parts = p.relative_to(ROOT).parts
    stage = next((STAGE[x] for x in parts if x in STAGE), "UNCLASSIFIED")
    if re.search(r"reverse|transfer|cross.?region", p.name + txt[:3000], re.I) and "S8" not in stage:
        stage += " (+ S8 cross-region transfer)"
    rows.append((f"`{p.relative_to(ROOT)}`", doc, "<br>".join(f"`{x}`" for x in ins) or "-",
                 "<br>".join(f"`{x}`" for x in outs) or "-", stage))
md_table(["File", "Purpose (first docstring/comment line)", "Main inputs", "Main outputs", "Stage"], rows)

# ---------------------------------------------------------------- PART 1.3 versions
w("## A.3 Python and package versions\n")
w(f"- Python: `{sys.version.split()[0]}` ({platform.platform()})")
pk = ["rasterio", "geopandas", "scikit-learn", "pysheds", "shap", "streamlit", "folium", "streamlit-folium",
      "numpy", "scipy", "earthengine-api", "pandas", "plotly", "matplotlib", "joblib", "fpdf2"]
rows = []
for n in pk:
    try:
        rows.append((n, metadata.version(n)))
    except metadata.PackageNotFoundError:
        rows.append((n, "NOT INSTALLED in this environment"))
md_table(["Package", "Installed version"], rows)
for f in ("requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml"):
    if (ROOT / f).exists():
        w(f"`{f}` found:\n```\n{(ROOT / f).read_text(errors='ignore')[:3000]}\n```\n")
    else:
        w(f"- `{f}`: NOT FOUND")
w()

# ---------------------------------------------------------------- PART 1.4 reproduce
w("## A.4 Reproduction material found in the repository\n")
for f in ("README.md", "Makefile", "run_all.sh"):
    if (ROOT / f).exists():
        w(f"`{f}` (first 80 lines):\n```\n" + "\n".join((ROOT / f).read_text(errors='ignore').splitlines()[:80]) + "\n```\n")
    else:
        w(f"- `{f}`: NOT FOUND")
w("\nScripts in folder order (use this as the command sequence; confirm each against the table in A.2):\n")
w("```bash")
for p in pyfiles:
    if "src" in p.relative_to(ROOT).parts and p.name != "__init__.py" and "dashboard" not in p.parts:
        w(f"python {p.relative_to(ROOT)}")
w("streamlit run src/dashboard/app_pro.py   # check which dashboard file is final")
w("```\n")
w("Git remote and history:\n```\n" + sh("git remote -v") + "\n" + sh("git log --oneline | head -20") + "\n```\n")

# ---------------------------------------------------------------- PART 1.5 config
w("## A.5 Hyperparameters and config values (every matching line, file:line)\n")
CFG = re.compile(r"RandomForestClassifier|n_estimators|max_depth|min_samples_leaf|min_samples_split|class_weight|"
                 r"random_state|\bseed\b|SEED|HAND_THRESHOLD|THRESH|threshold|TILE|BLOCK|block_size|checkerboard|"
                 r"accumulation|acc_|surrogate|n_jobs|max_features|N_BOOT|resample|HIGH_RISK|MEDIUM_RISK|buffer|"
                 r"max_samples|sample_size|n_samples|TARGET_YEARS", re.I)
w("```")
for p in pyfiles:
    for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
        if CFG.search(line) and not line.strip().startswith("#"):
            w(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:170]}")
w("```\n")

# ---------------------------------------------------------------- PART 1.6 code excerpts
w("## A.6 Core code excerpts (verbatim, numbered, max 40 lines each)\n")
w("Each excerpt starts a few lines before the first anchor match. Move the window if it misses the core logic.\n")
TARGETS = [("RF training and evaluation", ["train_rf.py"], r"RandomForestClassifier\("),
           ("HAND computation (check for buffered DEM use)", ["compute_hand.py"], r"flowdir|Grid\.from_raster|from_raster"),
           ("Exposure overlay", ["build_exposure_map.py"], r"flood_prone\s*=|exposure\s*="),
           ("Cross-region transfer evaluation", ["train_rf_reverse.py", "train_rf.py"], r"test_block|other|transfer|cross")]
for title, names, anchor in TARGETS:
    done = False
    for n in names:
        for p in [q for q in pyfiles if q.name == n]:
            lines = p.read_text(errors="ignore").splitlines()
            idx = next((i for i, l in enumerate(lines) if re.search(anchor, l)), None)
            if idx is None:
                continue
            a, b = max(0, idx - 8), min(len(lines), max(0, idx - 8) + 40)
            w(f"### {title}: `{p.relative_to(ROOT)}` lines {a+1}-{b}\n")
            w("```python")
            for k in range(a, b):
                w(f"{k+1:>4}  {lines[k]}")
            w("```\n")
            done = True
    if not done:
        w(f"### {title}: NOT FOUND (looked for {names} with pattern `{anchor}`)\n")

# ---------------------------------------------------------------- PART 2 outputs
w("# APPENDIX B MATERIAL (generated from the real outputs)\n")
w("## B.1 Every output file (png, json, csv, md; GeoTIFF layers listed separately)\n")
TAGS = [(r"builtup_classification_map", "Built-up classification map", "Used (Figures 4.2/4.3)"),
        (r"exposure_overlay", "Flood-exposure overlay", "Used (exposure figures), verify"),
        (r"shap_summary", "SHAP summary (surrogate)", "Used (Figure 4.5), verify"),
        (r"shap_exposed_vs_nonexposed", "SHAP, exposed vs non-exposed pixels", "NOT confirmed in thesis: appendix candidate"),
        (r"growth_flood_overlay", "New growth 2000-2025 by flood-proneness (yellow/red)", "Used (Figure 4.6a/b)"),
        (r"hand_nan_mask", "HAND NaN diagnostic before buffered DEM", "Used (Figures E.3/E.4)"),
        (r"dem_void_mask", "Raw DEM void diagnostic", "Check: likely Appendix E; verify"),
        (r"s2_preview", "Sentinel-2 true-colour QA composite", "Check in thesis; verify"),
        (r"confusion", "Confusion matrix", "Check in thesis; verify"),
        (r"transfer", "Transfer performance", "Check in thesis; verify"),
        (r"kpi", "Dashboard KPI values", "Source of Chapter 4 numbers"),
        (r"projection", "2030/2035 extrapolation", "Check in thesis; verify")]
rows = []
cands = []
for base in ("results", "outputs", "data/processed", "reports", "docs"):
    d = ROOT / base
    if d.exists():
        cands += [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".json", ".csv", ".md", ".txt", ".html"}]
for p in sorted(set(cands)):
    tag, status = next(((t, s) for pat, t, s in TAGS if re.search(pat, p.name)), ("UNLABELLED, inspect", "Check"))
    reg = "A" if "region_a" in p.name else "B" if "region_b" in p.name else "both/NA"
    rows.append((f"`{p.relative_to(ROOT)}`", tag, reg, f"{p.stat().st_size/1024:.0f} KB", status))
if rows:
    md_table(["File", "Shows", "Region", "Size", "Thesis status (from our chat; verify against your thesis)"], rows)
else:
    w("NOT FOUND: no results/, outputs/ or data/processed output files.\n")
tifs = sorted((ROOT / "data/processed").glob("*.tif")) if (ROOT / "data/processed").exists() else []
w("GeoTIFF layers in data/processed:\n")
md_table(["File", "Size"], [(f"`{t.relative_to(ROOT)}`", f"{t.stat().st_size/1e6:.1f} MB") for t in tifs] or [("NOT FOUND", "")])


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, f"{pre}{k}."))
        elif isinstance(v, list):
            o[f"{pre}{k}"] = (str(v)[:90] + "...") if len(str(v)) > 90 else str(v)
        else:
            o[f"{pre}{k}"] = round(v, 4) if isinstance(v, float) else v
    return o


w("## B.2 Contents of every JSON and CSV output (as clean tables)\n")
for p in sorted(set(c for c in cands if c.suffix in {".json", ".csv"})):
    w(f"### `{p.relative_to(ROOT)}`\n")
    try:
        if p.suffix == ".json":
            d = json.loads(p.read_text())
            if isinstance(d, dict) and all(isinstance(v, dict) for v in d.values()):
                fl = {k: flat(v) for k, v in d.items()}
                keys = sorted({kk for v in fl.values() for kk in v})
                md_table(["Field"] + list(fl), [[kk] + [fl[r].get(kk, "") for r in fl] for kk in keys])
            elif isinstance(d, dict):
                md_table(["Field", "Value"], list(flat(d).items()))
            else:
                w("```\n" + json.dumps(d)[:1500] + "\n```\n")
        else:
            with open(p, newline="") as f:
                rr = list(csv.reader(f))
            md_table(rr[0], rr[1:31])
            if len(rr) > 31:
                w(f"(showing 30 of {len(rr)-1} rows)\n")
    except Exception as e:
        w(f"(could not parse: {e})\n")

Path("appendix_material.md").write_text("\n".join(OUT))
print(f"Saved appendix_material.md ({len(OUT)} lines). Python files scanned: {len(pyfiles)}; output files found: {len(cands)}")
