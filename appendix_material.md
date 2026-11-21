# APPENDIX A MATERIAL (generated from the real project)

## A.1 Project folder structure (depth 3, data/.git/venv/__pycache__ ignored)

```
flood-housing-ai/
├── docs/
│   └── phase5_transferability_synthesis.md
├── models/
│   ├── rf_builtup_reduced_features.pkl
│   ├── rf_builtup_surrogate_shap.pkl
│   ├── rf_builtup_trainA.pkl
│   └── rf_builtup_trainB.pkl
├── notebooks/
├── results/
│   ├── figures/
│   │   ├── region_a_kelani_builtup_classification_map.png
│   │   ├── region_a_kelani_dem_void_mask.png
│   │   ├── region_a_kelani_exposure_overlay.png
│   │   ├── region_a_kelani_growth_flood_overlay.png
│   │   ├── region_a_kelani_hand_nan_mask.png
│   │   ├── region_a_kelani_s2_preview.png
│   │   ├── region_a_kelani_shap_exposed_vs_nonexposed.png
│   │   ├── region_a_kelani_shap_summary.png
│   │   ├── region_b_dakar_builtup_classification_map.png
│   │   ├── region_b_dakar_dem_void_mask.png
│   │   ├── region_b_dakar_exposure_overlay.png
│   │   ├── region_b_dakar_growth_flood_overlay.png
│   │   ├── region_b_dakar_hand_nan_mask.png
│   │   ├── region_b_dakar_s2_preview.png
│   │   ├── region_b_dakar_shap_exposed_vs_nonexposed.png
│   │   └── region_b_dakar_shap_summary.png
│   └── tables/
├── src/
│   ├── builtup_mapping/
│   │   ├── __init__.py
│   │   ├── train_rf.py
│   │   └── train_rf_reverse.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── regions.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── app_pro.py
│   │   ├── compute_kpis.py
│   │   └── compute_kpis_v2.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── run_shap_analysis.py
│   │   ├── shap_significance_test.py
│   │   └── train_surrogate.py
│   ├── exposure/
│   │   ├── __init__.py
│   │   ├── build_exposure_map.py
│   │   ├── render_builtup_maps.py
│   │   └── render_exposure_preview.py
│   ├── flood_layer/
│   │   ├── __init__.py
│   │   └── compute_hand.py
│   ├── ingestion/
│   │   ├── check_tasks.py
│   │   ├── export_dem_buffered.py
│   │   ├── export_ghsl.py
│   │   ├── export_open_buildings.py
│   │   ├── export_s2_with_swir.py
│   │   ├── export_temporal_snapshots.py
│   │   ├── export_to_drive.py
│   │   ├── qa_check.py
│   │   ├── render_previews.py
│   │   ├── test_ee_pull.py
│   │   ├── wait_for_buildings_v2.py
│   │   ├── wait_for_dem_buffered.py
│   │   ├── wait_for_ghsl.py
│   │   ├── wait_for_s2_full.py
│   │   ├── wait_for_tasks.py
│   │   └── wait_for_temporal.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── build_feature_stack.py
│   │   ├── fix_worldpop_nans.py
│   │   └── generate_labels.py
│   ├── transferability/
│   └── trend_modelling/
│       ├── __init__.py
│       ├── analyze_ghsl_trend.py
│       ├── apply_yearly_classifier.py
│       ├── apply_yearly_classifier_normalized.py
│       ├── build_temporal_features.py
│       ├── project_future_growth.py
│       └── render_growth_overlay.py
├── .gitignore
├── extract_appendix.py
└── requirements.txt
```

## A.2 Script inventory

Stage comes from the folder name. Inputs/outputs are found by scanning each file for path strings (heuristic: a path on a line that writes/saves/exports is listed as output). Check each row once.

| File | Purpose (first docstring/comment line) | Main inputs | Main outputs | Stage |
|---|---|---|---|---|
| `extract_appendix.py` | Collects facts for thesis Appendices A and B from the REAL project files. | `## A.1 Project folder structure (depth 3, data/.git/venv/__pycache__ ignored)\n`<br>`*.tif`<br>`.csv`<br>`.json`<br>`.png`<br>`GeoTIFF layers in data/processed:\n` | - | UNCLASSIFIED |
| `src/builtup_mapping/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S3 Random Forest built-up classification |
| `src/builtup_mapping/train_rf.py` | Trains a Random Forest built-up classifier on Region A (Kelani) using a | `\nModel saved to models/rf_builtup_trainA.pkl`<br>`data/processed`<br>`{region}_builtup_label.tif`<br>`{region}_feature_stack.tif` | `models/rf_builtup_trainA.pkl` | S3 Random Forest built-up classification (+ S8 cross-region transfer) |
| `src/builtup_mapping/train_rf_reverse.py` | Reverse-direction transferability check: trains on Region B (Dakar) using | `\nModel saved to models/rf_builtup_trainB.pkl`<br>`data/processed`<br>`{region}_builtup_label.tif`<br>`{region}_feature_stack.tif` | `models/rf_builtup_trainB.pkl` | S3 Random Forest built-up classification (+ S8 cross-region transfer) |
| `src/config/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | UNCLASSIFIED |
| `src/config/regions.py` | Bounding boxes for the two study regions. | - | - | UNCLASSIFIED |
| `src/dashboard/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | Streamlit dashboard |
| `src/dashboard/app.py` | Streamlit dashboard v2: flood-exposed housing detection system. | `dashboard_kpis.json`<br>`dashboard_summary.csv`<br>`data/processed`<br>`exposure_comparison.png`<br>`growth_by_year.csv`<br>`growth_comparison.png` | - | Streamlit dashboard |
| `src/dashboard/app_pro.py` | Professional FYP dashboard: Flood-Exposed Housing Detection System. | `dashboard_kpis.json`<br>`data/processed`<br>`data/raw`<br>`exposure_comparison.png`<br>`first to generate `data/processed/growth_projection_2030_2035.json`.`<br>`growth_projection_2030_2035.csv` | - | Streamlit dashboard |
| `src/dashboard/compute_kpis.py` | Precomputes all dashboard KPIs once and caches to JSON, so the Streamlit | `dashboard_kpis.json`<br>`data/processed`<br>`data/raw`<br>`{region}_builtup_label.tif`<br>`{region}_exposure_map.tif`<br>`{region}_feature_stack.tif` | - | Streamlit dashboard |
| `src/dashboard/compute_kpis_v2.py` | Extends dashboard_kpis.json with: | `\nSaved updated dashboard_kpis.json`<br>`{region}_builtup_label.tif`<br>`{region}_exposure_map.tif`<br>`{region}_feature_stack.tif` | `dashboard_kpis.json`<br>`data/processed` | Streamlit dashboard |
| `src/explainability/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S6 SHAP explainability |
| `src/explainability/run_shap_analysis.py` | SHAP explainability for the Region-A-trained RF built-up classifier. | `data/processed`<br>`models/rf_builtup_surrogate_shap.pkl`<br>`results/figures`<br>`results/figures/{region}_shap_exposed_vs_nonexposed.png`<br>`results/figures/{region}_shap_summary.png`<br>`{region}_builtup_label.tif` | - | S6 SHAP explainability (+ S8 cross-region transfer) |
| `src/explainability/shap_significance_test.py` | Bootstrap significance test: for each feature, is the mean \|SHAP\| | `data/processed`<br>`models/rf_builtup_surrogate_shap.pkl`<br>`{region}_builtup_label.tif`<br>`{region}_exposure_map.tif`<br>`{region}_feature_stack.tif` | - | S6 SHAP explainability |
| `src/explainability/train_surrogate.py` | Trains a shallow surrogate Random Forest (same features, same training | `\nSurrogate saved to models/rf_builtup_surrogate_shap.pkl`<br>`data/processed`<br>`models/rf_builtup_trainA.pkl`<br>`{region}_builtup_label.tif`<br>`{region}_feature_stack.tif` | `models/rf_builtup_surrogate_shap.pkl` | S6 SHAP explainability |
| `src/exposure/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S5 exposure analysis |
| `src/exposure/build_exposure_map.py` | Builds the flood exposure map for both regions: | `data/processed`<br>`{region}_builtup_label.tif`<br>`{region}_exposure_map.tif`<br>`{region}_feature_stack.tif`<br>`{region}_flood_prone_mask.tif` | - | S5 exposure analysis |
| `src/exposure/render_builtup_maps.py` | Renders clean built-up classification maps for Figures 4.2/4.3: | `data/processed`<br>`results/figures`<br>`results/figures/{region}_builtup_classification_map.png`<br>`{region}_builtup_label.tif`<br>`{region}_feature_stack.tif` | - | S5 exposure analysis |
| `src/exposure/render_exposure_preview.py` | NO DOCSTRING OR COMMENT FOUND | `data/processed`<br>`results/figures`<br>`results/figures/{region}_exposure_overlay.png`<br>`{region}_exposure_map.tif`<br>`{region}_feature_stack.tif`<br>`{region}_flood_prone_mask.tif` | - | S5 exposure analysis |
| `src/flood_layer/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S4 HAND flood-prone layer |
| `src/flood_layer/compute_hand.py` | Computes HAND using a BUFFERED DEM (extra ~5.5km margin) so D8 flow | `data/processed`<br>`data/raw`<br>`{region}_dem_buffered.tif`<br>`{region}_feature_stack.tif`<br>`{region}_hand_10m.tif`<br>`{region}_hand_buffered_tmp.tif` | - | S4 HAND flood-prone layer |
| `src/ingestion/check_tasks.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/export_dem_buffered.py` | Re-exports the DEM for both regions with a spatial buffer, so flow | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/export_ghsl.py` | Exports JRC Global Human Settlement Layer (GHSL) built-up surface data | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/export_open_buildings.py` | Re-export Google Open Buildings footprints for both regions, keeping only | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/export_s2_with_swir.py` | Re-export Sentinel-2 composites for both regions, adding B11 (SWIR), | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/export_temporal_snapshots.py` | Exports Sentinel-2 cloud-filtered median composites for 4 time points | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/export_to_drive.py` | Phase 0 / Step 4: submit Earth Engine batch export tasks to Google Drive | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/qa_check.py` | QA pass over all raw downloaded layers for both regions: | `data/raw`<br>`{region}_open_buildings_v2.shp`<br>`{region}_{layer}.tif` | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/render_previews.py` | Render quick true-color PNG previews of the Sentinel-2 composites for | `data/raw`<br>`results/figures`<br>`results/figures/{region}_s2_preview.png`<br>`{region}_s2_composite.tif` | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/test_ee_pull.py` | Sanity check: pull a SMALL cloud-filtered Sentinel-2 patch over the lower | `data/raw/region_a_kelani/s2_test_composite.tif` | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/wait_for_buildings_v2.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/wait_for_dem_buffered.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/wait_for_ghsl.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/wait_for_s2_full.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/wait_for_tasks.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/ingestion/wait_for_temporal.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S1 data ingestion (Earth Engine) |
| `src/preprocessing/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S2 preprocessing / feature stack |
| `src/preprocessing/build_feature_stack.py` | Phase 1 preprocessing: resample all layers onto the Sentinel-2 10m grid, | `data/processed`<br>`data/raw`<br>`{region}_dem.tif`<br>`{region}_feature_stack.tif`<br>`{region}_s1_composite.tif`<br>`{region}_s2_full.tif` | - | S2 preprocessing / feature stack |
| `src/preprocessing/fix_worldpop_nans.py` | Fixes WorldPop NaN gaps in the feature stacks: | `data/processed`<br>`{region}_feature_stack.tif` | - | S2 preprocessing / feature stack |
| `src/preprocessing/generate_labels.py` | Generates a binary built-up label raster (1=built-up, 0=not) aligned to | `data/processed`<br>`data/raw`<br>`{region}_builtup_label.tif`<br>`{region}_feature_stack.tif`<br>`{region}_open_buildings_v2.shp` | - | S2 preprocessing / feature stack |
| `src/trend_modelling/__init__.py` | NO DOCSTRING OR COMMENT FOUND | - | - | S7 GHSL growth trend and projection |
| `src/trend_modelling/analyze_ghsl_trend.py` | Analyzes GHSL built-up surface time series for both regions: | `data/processed`<br>`data/raw`<br>`{region}_feature_stack.tif`<br>`{region}_flood_prone_mask.tif`<br>`{region}_ghsl_{epoch}.tif` | - | S7 GHSL growth trend and projection |
| `src/trend_modelling/apply_yearly_classifier.py` | Memory-conservative version: applies the already-trained reduced-feature | `data/processed`<br>`data/raw`<br>`models/rf_builtup_reduced_features.pkl`<br>`{region}_builtup_{year}.tif`<br>`{region}_feature_stack.tif`<br>`{region}_s2_{year}.tif` | - | S7 GHSL growth trend and projection |
| `src/trend_modelling/apply_yearly_classifier_normalized.py` | Fixes cross-year radiometric drift by percentile-matching each year's | `data/processed`<br>`data/raw`<br>`models/rf_builtup_reduced_features.pkl`<br>`{region}_builtup_{year}_norm.tif`<br>`{region}_feature_stack.tif`<br>`{region}_s2_{REFERENCE_YEAR}.tif` | - | S7 GHSL growth trend and projection |
| `src/trend_modelling/build_temporal_features.py` | For each region and year, computes NDVI/NDBI from the temporal Sentinel-2 | `data/processed`<br>`data/raw`<br>`region_a_kelani_builtup_label.tif`<br>`region_a_kelani_feature_stack.tif`<br>`{region}_builtup_{year}.tif`<br>`{region}_feature_stack.tif` | `models/rf_builtup_reduced_features.pkl` | S7 GHSL growth trend and projection |
| `src/trend_modelling/project_future_growth.py` | --- Phase 4 / Step 6: illustrative linear extrapolation of GHSL trend to future years --- | `\nSaved data/processed/growth_projection_2030_2035.json`<br>`dashboard_kpis.json` | `data/processed`<br>`growth_projection_2030_2035.json` | S7 GHSL growth trend and projection |
| `src/trend_modelling/render_growth_overlay.py` | NO DOCSTRING OR COMMENT FOUND | `data/processed`<br>`data/raw`<br>`results/figures`<br>`results/figures/{region}_growth_flood_overlay.png`<br>`{region}_feature_stack.tif`<br>`{region}_flood_prone_mask.tif` | - | S7 GHSL growth trend and projection |

## A.3 Python and package versions

- Python: `3.10.12` (Linux-6.8.0-138-generic-x86_64-with-glibc2.35)
| Package | Installed version |
|---|---|
| rasterio | 1.4.4 |
| geopandas | 1.1.4 |
| scikit-learn | 1.7.2 |
| pysheds | 0.5 |
| shap | 0.49.1 |
| streamlit | 1.59.2 |
| folium | 0.20.0 |
| streamlit-folium | 0.26.2 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| earthengine-api | 1.7.35 |
| pandas | 2.3.3 |
| plotly | 6.9.0 |
| matplotlib | 3.10.9 |
| joblib | 1.5.3 |
| fpdf2 | 2.8.7 |

`requirements.txt` found:
```
# Geospatial I/O
gdal==$(gdal-config --version)
rasterio>=1.3
geopandas>=0.14
shapely>=2.0
pyproj>=3.6
fiona>=1.9

# Earth Engine (Sentinel-2/1 access)
earthengine-api>=0.1.400
geemap>=0.32

# ML / stats
scikit-learn>=1.4
torch>=2.2
torchvision>=0.17
shap>=0.45
xgboost>=2.0
scipy>=1.12
numpy>=1.26
pandas>=2.2

# Raster/vector processing helpers
rasterstats>=0.19
scikit-image>=0.22
opencv-python-headless>=4.9

# Visualization / dashboard
matplotlib>=3.8
seaborn>=0.13
streamlit>=1.32
folium>=0.15

# Notebook support
jupyter>=1.0
ipykernel>=6.29

# Utilities
python-dotenv>=1.0
tqdm>=4.66

```

- `environment.yml`: NOT FOUND
- `environment.yaml`: NOT FOUND
- `pyproject.toml`: NOT FOUND

## A.4 Reproduction material found in the repository

- `README.md`: NOT FOUND
- `Makefile`: NOT FOUND
- `run_all.sh`: NOT FOUND

Scripts in folder order (use this as the command sequence; confirm each against the table in A.2):

```bash
python src/builtup_mapping/train_rf.py
python src/builtup_mapping/train_rf_reverse.py
python src/config/regions.py
python src/explainability/run_shap_analysis.py
python src/explainability/shap_significance_test.py
python src/explainability/train_surrogate.py
python src/exposure/build_exposure_map.py
python src/exposure/render_builtup_maps.py
python src/exposure/render_exposure_preview.py
python src/flood_layer/compute_hand.py
python src/ingestion/check_tasks.py
python src/ingestion/export_dem_buffered.py
python src/ingestion/export_ghsl.py
python src/ingestion/export_open_buildings.py
python src/ingestion/export_s2_with_swir.py
python src/ingestion/export_temporal_snapshots.py
python src/ingestion/export_to_drive.py
python src/ingestion/qa_check.py
python src/ingestion/render_previews.py
python src/ingestion/test_ee_pull.py
python src/ingestion/wait_for_buildings_v2.py
python src/ingestion/wait_for_dem_buffered.py
python src/ingestion/wait_for_ghsl.py
python src/ingestion/wait_for_s2_full.py
python src/ingestion/wait_for_tasks.py
python src/ingestion/wait_for_temporal.py
python src/preprocessing/build_feature_stack.py
python src/preprocessing/fix_worldpop_nans.py
python src/preprocessing/generate_labels.py
python src/trend_modelling/analyze_ghsl_trend.py
python src/trend_modelling/apply_yearly_classifier.py
python src/trend_modelling/apply_yearly_classifier_normalized.py
python src/trend_modelling/build_temporal_features.py
python src/trend_modelling/project_future_growth.py
python src/trend_modelling/render_growth_overlay.py
streamlit run src/dashboard/app_pro.py   # check which dashboard file is final
```

Git remote and history:
```
origin	https://github.com/EHMDIOUF/flood-housing-ai.git (fetch)
origin	https://github.com/EHMDIOUF/flood-housing-ai.git (push)
5482a03 Phase 6: Streamlit dashboard (executive summary, exposure map, growth trends, transferability, risk assessment) + rendered built-up classification maps
6adcb2b Phase 5: cross-region transferability synthesis writeup
a814521 Phase 4: GHSL-based expansion trend analysis + flood-prone growth cluster identification (superseded noisy classifier-based approach)
9f397ca Phase 3: SHAP explainability via surrogate model + bootstrap significance test on exposed vs non-exposed built-up
c68c9b3 Phase 2: flood-prone mask (HAND<5m) + built-up exposure overlay, visually validated
4f99119 Phase 2: HAND flood-proneness layer (buffered DEM fix for boundary drainage artifacts)
645488b Phase 1: symmetric cross-region transferability check (A->B and B->A)
0ae8423 Phase 1: RF built-up classifier baseline (same-region vs cross-region transfer)
```

## A.5 Hyperparameters and config values (every matching line, file:line)

```
extract_appendix.py:119: CFG = re.compile(r"RandomForestClassifier|n_estimators|max_depth|min_samples_leaf|min_samples_split|class_weight|"
extract_appendix.py:120: r"random_state|\bseed\b|SEED|HAND_THRESHOLD|THRESH|threshold|TILE|BLOCK|block_size|checkerboard|"
extract_appendix.py:121: r"accumulation|acc_|surrogate|n_jobs|max_features|N_BOOT|resample|HIGH_RISK|MEDIUM_RISK|buffer|"
extract_appendix.py:122: r"max_samples|sample_size|n_samples|TARGET_YEARS", re.I)
extract_appendix.py:133: TARGETS = [("RF training and evaluation", ["train_rf.py"], r"RandomForestClassifier\("),
extract_appendix.py:134: ("HAND computation (check for buffered DEM use)", ["compute_hand.py"], r"flowdir|Grid\.from_raster|from_raster"),
extract_appendix.py:136: ("Cross-region transfer evaluation", ["train_rf_reverse.py", "train_rf.py"], r"test_block|other|transfer|cross")]
extract_appendix.py:160: (r"shap_summary", "SHAP summary (surrogate)", "Used (Figure 4.5), verify"),
extract_appendix.py:163: (r"hand_nan_mask", "HAND NaN diagnostic before buffered DEM", "Used (Figures E.3/E.4)"),
src/builtup_mapping/train_rf.py:3: CHECKERBOARD spatial split (avoids the distribution-shift problem of a
src/builtup_mapping/train_rf.py:5: 1. Held-out checkerboard tiles within Region A (same-region baseline)
src/builtup_mapping/train_rf.py:15: from sklearn.ensemble import RandomForestClassifier
src/builtup_mapping/train_rf.py:27: TILE_SIZE = 64  # pixels; checkerboard cell size
src/builtup_mapping/train_rf.py:43: def checkerboard_mask(H, W, tile_size):
src/builtup_mapping/train_rf.py:44: """True = train tile, False = test tile, in a checkerboard pattern."""
src/builtup_mapping/train_rf.py:45: row_tile = (np.arange(H) // tile_size)[:, None]
src/builtup_mapping/train_rf.py:46: col_tile = (np.arange(W) // tile_size)[None, :]
src/builtup_mapping/train_rf.py:47: return ((row_tile + col_tile) % 2 == 0)
src/builtup_mapping/train_rf.py:82: train_mask = checkerboard_mask(H, W, TILE_SIZE)
src/builtup_mapping/train_rf.py:92: clf = RandomForestClassifier(
src/builtup_mapping/train_rf.py:93: n_estimators=200, max_depth=20, min_samples_leaf=5,
src/builtup_mapping/train_rf.py:94: n_jobs=-1, random_state=42, class_weight="balanced",
src/builtup_mapping/train_rf.py:98: evaluate("Region A held-out checkerboard tiles (same-region baseline)", clf, X_test_a, y_test_a)
src/builtup_mapping/train_rf_reverse.py:3: a checkerboard spatial split, evaluates on B's own held-out tiles
src/builtup_mapping/train_rf_reverse.py:10: from sklearn.ensemble import RandomForestClassifier
src/builtup_mapping/train_rf_reverse.py:22: TILE_SIZE = 64
src/builtup_mapping/train_rf_reverse.py:38: def checkerboard_mask(H, W, tile_size):
src/builtup_mapping/train_rf_reverse.py:39: row_tile = (np.arange(H) // tile_size)[:, None]
src/builtup_mapping/train_rf_reverse.py:40: col_tile = (np.arange(W) // tile_size)[None, :]
src/builtup_mapping/train_rf_reverse.py:41: return ((row_tile + col_tile) % 2 == 0)
src/builtup_mapping/train_rf_reverse.py:76: train_mask = checkerboard_mask(H, W, TILE_SIZE)
src/builtup_mapping/train_rf_reverse.py:86: clf = RandomForestClassifier(
src/builtup_mapping/train_rf_reverse.py:87: n_estimators=200, max_depth=20, min_samples_leaf=5,
src/builtup_mapping/train_rf_reverse.py:88: n_jobs=-1, random_state=42, class_weight="balanced",
src/builtup_mapping/train_rf_reverse.py:92: evaluate("Region B held-out checkerboard tiles (same-region baseline)", clf, X_test_b, y_test_b)
src/dashboard/app.py:57: lo, hi = np.percentile(band, [low, high])
src/dashboard/app.py:245: indices, and elevation, validated via a spatially-blocked
src/dashboard/app.py:246: checkerboard train/test split.
src/dashboard/app.py:248: computed via D8 flow routing on a buffered Copernicus GLO-30 DEM,
src/dashboard/app.py:249: thresholded at 5m per literature convention.
src/dashboard/app.py:252: 4. **Explainability**: SHAP analysis via a depth-limited surrogate
src/dashboard/app.py:281: - HAND computation required a buffered DEM re-export to resolve boundary
src/dashboard/app_pro.py:86: lo, hi = np.percentile(band, [low, high])
src/dashboard/app_pro.py:104: def load_and_resample_ghsl(epoch):
src/dashboard/app_pro.py:116: ghsl_2000 = load_and_resample_ghsl(2000)
src/dashboard/app_pro.py:117: ghsl_2025 = load_and_resample_ghsl(2025)
src/dashboard/app_pro.py:265: m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
src/dashboard/app_pro.py:314: gmap = folium.Map(location=gcenter, zoom_start=12, tiles="OpenStreetMap")
src/dashboard/app_pro.py:515: "SHAP analysis (via a depth-limited surrogate model) shows the two region-specific "
src/dashboard/app_pro.py:557: "Resolution": "Re-computed on buffered DEM (+5.5km margin); reduced to <5% NaN"},
src/dashboard/app_pro.py:575: and elevation; validated via spatially-blocked checkerboard train/test split.
src/dashboard/app_pro.py:576: 3. **Flood-Proneness (HAND)** — D8 flow routing on a buffered DEM; thresholded at 5m.
src/dashboard/app_pro.py:578: 5. **Explainability** — SHAP via a depth-limited surrogate model for tractable exact computation.
src/dashboard/compute_kpis_v2.py:3: - Risk tiers (High/Medium/Low) based on HAND thresholds, as % of built-up area
src/dashboard/compute_kpis_v2.py:15: HIGH_RISK_MAX = 2.0
src/dashboard/compute_kpis_v2.py:16: MEDIUM_RISK_MAX = 5.0
src/dashboard/compute_kpis_v2.py:38: high_risk = builtup_land & (hand < HIGH_RISK_MAX)
src/dashboard/compute_kpis_v2.py:39: medium_risk = builtup_land & (hand >= HIGH_RISK_MAX) & (hand < MEDIUM_RISK_MAX)
src/dashboard/compute_kpis_v2.py:40: low_risk = builtup_land & (hand >= MEDIUM_RISK_MAX)
src/dashboard/compute_kpis_v2.py:42: pct_high = 100 * high_risk.sum() / n_builtup
src/dashboard/compute_kpis_v2.py:43: pct_medium = 100 * medium_risk.sum() / n_builtup
src/explainability/run_shap_analysis.py:21: SAMPLE_SIZE = 4000
src/explainability/run_shap_analysis.py:74: clf = joblib.load("models/rf_builtup_surrogate_shap.pkl")
src/explainability/run_shap_analysis.py:80: X_sample, y_sample, group_labels = stratified_sample(features, label, water, exposure, SAMPLE_SIZE)
src/explainability/shap_significance_test.py:6: Resamples both groups with replacement many times, recomputes the mean
src/explainability/shap_significance_test.py:21: SAMPLE_SIZE = 4000
src/explainability/shap_significance_test.py:22: N_BOOTSTRAP = 2000
src/explainability/shap_significance_test.py:58: print("Loading surrogate model...")
src/explainability/shap_significance_test.py:59: clf = joblib.load("models/rf_builtup_surrogate_shap.pkl")
src/explainability/shap_significance_test.py:66: X_exposed = sample_group(features, label, water, exposure, "exposed", SAMPLE_SIZE)
src/explainability/shap_significance_test.py:67: X_nonexposed = sample_group(features, label, water, exposure, "nonexposed", SAMPLE_SIZE)
src/explainability/shap_significance_test.py:78: print(f"  Bootstrapping ({N_BOOTSTRAP} resamples)...")
src/explainability/shap_significance_test.py:80: boot_diffs = np.zeros((N_BOOTSTRAP, len(FEATURE_BANDS)))
src/explainability/shap_significance_test.py:81: for b in range(N_BOOTSTRAP):
src/explainability/shap_significance_test.py:87: ci_low = np.percentile(boot_diffs, 2.5, axis=0)
src/explainability/shap_significance_test.py:88: ci_high = np.percentile(boot_diffs, 97.5, axis=0)
src/explainability/train_surrogate.py:2: Trains a shallow surrogate Random Forest (same features, same training
src/explainability/train_surrogate.py:4: The production model (max_depth=20, ~43k leaves/tree) is too large for
src/explainability/train_surrogate.py:5: tractable exact TreeSHAP. This surrogate uses max_depth=8, which should
src/explainability/train_surrogate.py:9: Reports: surrogate's own accuracy on the same held-out test sets used
src/explainability/train_surrogate.py:12: matter for trusting the surrogate's explanations as representative.
src/explainability/train_surrogate.py:16: from sklearn.ensemble import RandomForestClassifier
src/explainability/train_surrogate.py:25: TILE_SIZE = 64
src/explainability/train_surrogate.py:41: def checkerboard_mask(H, W, tile_size):
src/explainability/train_surrogate.py:42: row_tile = (np.arange(H) // tile_size)[:, None]
src/explainability/train_surrogate.py:43: col_tile = (np.arange(W) // tile_size)[None, :]
src/explainability/train_surrogate.py:44: return ((row_tile + col_tile) % 2 == 0)
src/explainability/train_surrogate.py:61: train_mask = checkerboard_mask(H, W, TILE_SIZE)
src/explainability/train_surrogate.py:67: print("Training shallow surrogate Random Forest (max_depth=8)...")
src/explainability/train_surrogate.py:68: surrogate = RandomForestClassifier(
src/explainability/train_surrogate.py:69: n_estimators=200, max_depth=8, min_samples_leaf=20,
src/explainability/train_surrogate.py:70: n_jobs=-1, random_state=42, class_weight="balanced",
src/explainability/train_surrogate.py:72: surrogate.fit(X_train, y_train)
src/explainability/train_surrogate.py:74: leaf_counts = [tree.tree_.n_leaves for tree in surrogate.estimators_]
src/explainability/train_surrogate.py:75: print(f"Surrogate leaf count per tree: min={min(leaf_counts)} max={max(leaf_counts)} mean={sum(leaf_counts)/len(leaf_counts):.0f}")
src/explainability/train_surrogate.py:78: pred_a = surrogate.predict(X_test_a)
src/explainability/train_surrogate.py:79: print(f"\nSurrogate accuracy on Region A held-out:")
src/explainability/train_surrogate.py:86: pred_b = surrogate.predict(X_test_b)
src/explainability/train_surrogate.py:87: print(f"Surrogate accuracy on Region B (transfer):")
src/explainability/train_surrogate.py:98: print(f"\nSurrogate vs production agreement rate:")
src/explainability/train_surrogate.py:103: joblib.dump(surrogate, "models/rf_builtup_surrogate_shap.pkl")
src/explainability/train_surrogate.py:104: print("\nSurrogate saved to models/rf_builtup_surrogate_shap.pkl")
src/exposure/build_exposure_map.py:3: 1. Flood-prone mask: HAND < 5m (literature-standard threshold)
src/exposure/build_exposure_map.py:13: HAND_THRESHOLD_M = 5.0
src/exposure/build_exposure_map.py:30: flood_prone = (hand < HAND_THRESHOLD_M).astype(np.uint8)
src/exposure/build_exposure_map.py:35: print(f"  Flood-prone (HAND < {HAND_THRESHOLD_M}m) land pixels: {flood_prone_pixels} "
src/exposure/render_exposure_preview.py:26: lo, hi = np.percentile(band, [low, high])
src/flood_layer/compute_hand.py:2: Computes HAND using a BUFFERED DEM (extra ~5.5km margin) so D8 flow
src/flood_layer/compute_hand.py:3: routing has real upstream/downstream context, avoiding the tile-boundary
src/flood_layer/compute_hand.py:4: NaN artifact seen when computing directly on the unbuffered crop. The
src/flood_layer/compute_hand.py:6: box before being resampled onto the 10m feature-stack grid.
src/flood_layer/compute_hand.py:18: ACC_THRESHOLD = 500
src/flood_layer/compute_hand.py:22: dem_path = Path("data/raw") / region / f"{region}_dem_buffered.tif"
src/flood_layer/compute_hand.py:23: orig_bbox = cfg["bbox"]  # [minlon, minlat, maxlon, maxlat] -- ORIGINAL (unbuffered) AOI
src/flood_layer/compute_hand.py:36: print("  Computing flow accumulation...")
src/flood_layer/compute_hand.py:37: acc = grid.accumulation(fdir)
src/flood_layer/compute_hand.py:39: print(f"  Extracting stream network (accumulation > {ACC_THRESHOLD})...")
src/flood_layer/compute_hand.py:40: streams = (acc > ACC_THRESHOLD)
src/flood_layer/compute_hand.py:42: print(f"  Stream cells identified: {n_stream_cells} ({100*n_stream_cells/streams.size:.3f}% of buffered area)")
src/flood_layer/compute_hand.py:44: print("  Computing HAND on buffered DEM...")
src/flood_layer/compute_hand.py:50: n_nan_buffered = np.isnan(hand_array).sum()
src/flood_layer/compute_hand.py:51: print(f"  NaN cells in buffered HAND: {n_nan_buffered} ({100*n_nan_buffered/hand_array.size:.2f}%)")
src/flood_layer/compute_hand.py:54: buffered_profile = {
src/flood_layer/compute_hand.py:58: tmp_path = Path("data/processed") / f"{region}_hand_buffered_tmp.tif"
src/flood_layer/compute_hand.py:60: with rasterio.open(tmp_path, "w", **buffered_profile) as dst:
src/flood_layer/compute_hand.py:113: tmp_path.unlink()  # cleanup temp buffered file
src/ingestion/export_dem_buffered.py:2: Re-exports the DEM for both regions with a spatial buffer, so flow
src/ingestion/export_dem_buffered.py:4: (avoids the D8-flow-exits-tile-boundary NaN artifact in HAND).
src/ingestion/export_dem_buffered.py:15: BUFFER_DEG = 0.05  # ~5.5km at these latitudes
src/ingestion/export_dem_buffered.py:21: buffered_bbox = [
src/ingestion/export_dem_buffered.py:22: minlon - BUFFER_DEG, minlat - BUFFER_DEG,
src/ingestion/export_dem_buffered.py:23: maxlon + BUFFER_DEG, maxlat + BUFFER_DEG,
src/ingestion/export_dem_buffered.py:25: geom = ee.Geometry.Rectangle(buffered_bbox)
src/ingestion/export_dem_buffered.py:31: description=f"{region_key}_dem_buffered",
src/ingestion/export_dem_buffered.py:39: print(f"submitted: {region_key}_dem_buffered (id={task.id})  bbox={buffered_bbox}")
src/ingestion/export_dem_buffered.py:42: print(f"\nSubmitted {len(submitted)} buffered DEM export tasks.")
src/ingestion/render_previews.py:23: lo, hi = np.percentile(band, [low, high])
src/ingestion/wait_for_dem_buffered.py:3: TARGETS = ["region_a_kelani_dem_buffered", "region_b_dakar_dem_buffered"]
src/ingestion/wait_for_dem_buffered.py:12: print("\nBoth buffered DEM exports completed.")
src/preprocessing/build_feature_stack.py:2: Phase 1 preprocessing: resample all layers onto the Sentinel-2 10m grid,
src/preprocessing/build_feature_stack.py:20: def resample_to_reference(src_path, ref_profile, resampling_method):
src/preprocessing/build_feature_stack.py:21: """Read src_path and resample it onto the reference grid/profile."""
src/preprocessing/build_feature_stack.py:57: s1_vv = resample_to_reference(s1_path, ref_profile, Resampling.bilinear) if src.count >= 1 else None
src/preprocessing/build_feature_stack.py:62: def resample_band(path, band_idx, method):
src/preprocessing/build_feature_stack.py:76: s1_vv = resample_band(s1_path, 1, Resampling.bilinear)
src/preprocessing/build_feature_stack.py:77: s1_vh = resample_band(s1_path, 2, Resampling.bilinear)
src/preprocessing/build_feature_stack.py:81: dem = resample_band(dem_path, 1, Resampling.bilinear)
src/preprocessing/build_feature_stack.py:85: worldcover = resample_band(wc_path, 1, Resampling.nearest)
src/preprocessing/build_feature_stack.py:89: worldpop = resample_band(wp_path, 1, Resampling.bilinear)
src/trend_modelling/analyze_ghsl_trend.py:6: 2. Resamples each epoch's GHSL layer onto the 10m feature-stack grid,
src/trend_modelling/analyze_ghsl_trend.py:50: def load_and_resample(epoch):
src/trend_modelling/analyze_ghsl_trend.py:64: ghsl_first = load_and_resample(EPOCHS[0])
src/trend_modelling/analyze_ghsl_trend.py:65: ghsl_last = load_and_resample(EPOCHS[-1])
src/trend_modelling/analyze_ghsl_trend.py:69: BUILTUP_M2_THRESHOLD = 1000
src/trend_modelling/analyze_ghsl_trend.py:70: builtup_first = ghsl_first > BUILTUP_M2_THRESHOLD
src/trend_modelling/analyze_ghsl_trend.py:71: builtup_last = ghsl_last > BUILTUP_M2_THRESHOLD
src/trend_modelling/analyze_ghsl_trend.py:99: print("  No new growth pixels detected between first and last epoch at this threshold.")
src/trend_modelling/apply_yearly_classifier.py:45: resampled = np.empty((len(REDUCED_FEATURES), ref_h, ref_w), dtype=np.float32)
src/trend_modelling/apply_yearly_classifier.py:48: source=band_arr, destination=resampled[i],
src/trend_modelling/apply_yearly_classifier.py:57: X_yr = resampled.reshape(len(REDUCED_FEATURES), -1).T
src/trend_modelling/apply_yearly_classifier.py:58: del resampled
src/trend_modelling/apply_yearly_classifier_normalized.py:2: Fixes cross-year radiometric drift by percentile-matching each year's
src/trend_modelling/apply_yearly_classifier_normalized.py:35: ref_percentiles = {b: np.percentile(arr, [2, 98]) for b, arr in zip(RAW_BANDS, ref_bands)}
src/trend_modelling/apply_yearly_classifier_normalized.py:36: print(f"  Reference ({REFERENCE_YEAR}) percentiles: " +
src/trend_modelling/apply_yearly_classifier_normalized.py:37: "  ".join(f"{b}=[{p[0]:.0f},{p[1]:.0f}]" for b, p in ref_percentiles.items()))
src/trend_modelling/apply_yearly_classifier_normalized.py:52: lo_src, hi_src = np.percentile(raw[b], [2, 98])
src/trend_modelling/apply_yearly_classifier_normalized.py:53: lo_ref, hi_ref = ref_percentiles[b]
src/trend_modelling/apply_yearly_classifier_normalized.py:63: resampled = np.empty((len(REDUCED_FEATURES), ref_h, ref_w), dtype=np.float32)
src/trend_modelling/apply_yearly_classifier_normalized.py:68: source=band_arr, destination=resampled[i],
src/trend_modelling/apply_yearly_classifier_normalized.py:76: X_yr = resampled.reshape(len(REDUCED_FEATURES), -1).T
src/trend_modelling/apply_yearly_classifier_normalized.py:77: del resampled
src/trend_modelling/build_temporal_features.py:4: Phase 1) to produce a per-year built-up map, resampled onto the same
src/trend_modelling/build_temporal_features.py:14: from sklearn.ensemble import RandomForestClassifier
src/trend_modelling/build_temporal_features.py:22: TILE_SIZE = 64
src/trend_modelling/build_temporal_features.py:25: def checkerboard_mask(H, W, tile_size):
src/trend_modelling/build_temporal_features.py:26: row_tile = (np.arange(H) // tile_size)[:, None]
src/trend_modelling/build_temporal_features.py:27: col_tile = (np.arange(W) // tile_size)[None, :]
src/trend_modelling/build_temporal_features.py:28: return ((row_tile + col_tile) % 2 == 0)
src/trend_modelling/build_temporal_features.py:57: train_mask = checkerboard_mask(H, W, TILE_SIZE)
src/trend_modelling/build_temporal_features.py:65: clf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5,
src/trend_modelling/build_temporal_features.py:66: n_jobs=-1, random_state=42, class_weight="balanced")
src/trend_modelling/build_temporal_features.py:93: resampled = np.empty((len(REDUCED_FEATURES), ref_profile["height"], ref_profile["width"]), dtype=np.float32)
src/trend_modelling/build_temporal_features.py:96: source=reduced_yr[i], destination=resampled[i],
src/trend_modelling/build_temporal_features.py:102: X_yr = resampled.reshape(len(REDUCED_FEATURES), -1).T
src/trend_modelling/project_future_growth.py:10: TARGET_YEARS = [2030, 2035]
src/trend_modelling/project_future_growth.py:27: for ty in TARGET_YEARS:
src/trend_modelling/render_growth_overlay.py:19: def load_and_resample(epoch):
src/trend_modelling/render_growth_overlay.py:30: ghsl_2000 = load_and_resample(2000)
src/trend_modelling/render_growth_overlay.py:31: ghsl_2025 = load_and_resample(2025)
src/trend_modelling/render_growth_overlay.py:40: lo, hi = np.percentile(band, [low, high])
```

## A.6 Core code excerpts (verbatim, numbered, max 40 lines each)

Each excerpt starts a few lines before the first anchor match. Move the window if it misses the core logic.

### RF training and evaluation: `src/builtup_mapping/train_rf.py` lines 84-113

```python
  84  
  85  X_train, y_train = to_pixel_table(feat_a, label_a, water_a, mask_extra=train_mask)
  86  X_test_a, y_test_a = to_pixel_table(feat_a, label_a, water_a, mask_extra=test_mask)
  87  
  88  print(f"Train pixels: {len(y_train)}  (built-up rate: {y_train.mean():.4f})")
  89  print(f"Held-out Region A test pixels: {len(y_test_a)}  (built-up rate: {y_test_a.mean():.4f})")
  90  
  91  print("\nTraining Random Forest...")
  92  clf = RandomForestClassifier(
  93      n_estimators=200, max_depth=20, min_samples_leaf=5,
  94      n_jobs=-1, random_state=42, class_weight="balanced",
  95  )
  96  clf.fit(X_train, y_train)
  97  
  98  evaluate("Region A held-out checkerboard tiles (same-region baseline)", clf, X_test_a, y_test_a)
  99  
 100  print("\nLoading Region B (cross-region transfer test)...")
 101  feat_b, label_b, water_b = load_region("region_b_dakar")
 102  X_test_b, y_test_b = to_pixel_table(feat_b, label_b, water_b)
 103  evaluate("Region B full (cross-region transfer)", clf, X_test_b, y_test_b)
 104  
 105  Path("models").mkdir(exist_ok=True)
 106  joblib.dump(clf, "models/rf_builtup_trainA.pkl")
 107  
 108  importances = clf.feature_importances_
 109  print("\n--- Feature importances ---")
 110  for name, imp in sorted(zip(FEATURE_BANDS, importances), key=lambda x: -x[1]):
 111      print(f"  {name:8s} {imp:.4f}")
 112  
 113  print("\nModel saved to models/rf_builtup_trainA.pkl")
```

### HAND computation (check for buffered DEM use): `src/flood_layer/compute_hand.py` lines 17-56

```python
  17  
  18  ACC_THRESHOLD = 500
  19  
  20  for region, cfg in REGIONS.items():
  21      print(f"\n=== {region} ===")
  22      dem_path = Path("data/raw") / region / f"{region}_dem_buffered.tif"
  23      orig_bbox = cfg["bbox"]  # [minlon, minlat, maxlon, maxlat] -- ORIGINAL (unbuffered) AOI
  24  
  25      grid = Grid.from_raster(str(dem_path))
  26      dem = grid.read_raster(str(dem_path))
  27  
  28      print("  Filling pits and depressions...")
  29      pit_filled = grid.fill_pits(dem)
  30      flooded = grid.fill_depressions(pit_filled)
  31      inflated = grid.resolve_flats(flooded)
  32  
  33      print("  Computing D8 flow direction...")
  34      fdir = grid.flowdir(inflated)
  35  
  36      print("  Computing flow accumulation...")
  37      acc = grid.accumulation(fdir)
  38  
  39      print(f"  Extracting stream network (accumulation > {ACC_THRESHOLD})...")
  40      streams = (acc > ACC_THRESHOLD)
  41      n_stream_cells = int(streams.sum())
  42      print(f"  Stream cells identified: {n_stream_cells} ({100*n_stream_cells/streams.size:.3f}% of buffered area)")
  43  
  44      print("  Computing HAND on buffered DEM...")
  45      hand = grid.compute_hand(fdir, inflated, streams)
  46      hand_array = np.asarray(hand, dtype=np.float32)
  47      affine = grid.affine
  48      crs = grid.crs
  49  
  50      n_nan_buffered = np.isnan(hand_array).sum()
  51      print(f"  NaN cells in buffered HAND: {n_nan_buffered} ({100*n_nan_buffered/hand_array.size:.2f}%)")
  52  
  53      # --- Write buffered HAND to a temp raster (needed so we can crop with rasterio) ---
  54      buffered_profile = {
  55          "driver": "GTiff", "height": hand_array.shape[0], "width": hand_array.shape[1],
  56          "count": 1, "dtype": "float32", "crs": crs, "transform": affine, "nodata": -9999.0,
```

### Exposure overlay: `src/exposure/build_exposure_map.py` lines 22-61

```python
  22          names = list(src.descriptions)
  23          hand = src.read(names.index("HAND") + 1)
  24          water_mask = src.read(names.index("WATER_MASK") + 1)
  25  
  26      with rasterio.open(label_path) as src:
  27          builtup = src.read(1)
  28  
  29      # --- Flood-prone mask (exclude water pixels -- flood-proneness of open water is meaningless) ---
  30      flood_prone = (hand < HAND_THRESHOLD_M).astype(np.uint8)
  31      flood_prone[water_mask == 1] = 0
  32  
  33      land_pixels = (water_mask == 0).sum()
  34      flood_prone_pixels = flood_prone.sum()
  35      print(f"  Flood-prone (HAND < {HAND_THRESHOLD_M}m) land pixels: {flood_prone_pixels} "
  36            f"({100*flood_prone_pixels/land_pixels:.2f}% of land area)")
  37  
  38      # --- Exposure: flood-prone AND built-up ---
  39      exposure = ((flood_prone == 1) & (builtup == 1)).astype(np.uint8)
  40      builtup_land_pixels = ((builtup == 1) & (water_mask == 0)).sum()
  41      exposure_pixels = exposure.sum()
  42  
  43      print(f"  Built-up land pixels: {builtup_land_pixels}")
  44      print(f"  Exposed (flood-prone + built-up) pixels: {exposure_pixels} "
  45            f"({100*exposure_pixels/builtup_land_pixels:.2f}% of built-up area is flood-exposed)")
  46  
  47      # --- Save flood-prone mask ---
  48      out_profile = profile.copy()
  49      out_profile.update(count=1, dtype="uint8", compress="lzw")
  50  
  51      fp_path = Path("data/processed") / f"{region}_flood_prone_mask.tif"
  52      with rasterio.open(fp_path, "w", **out_profile) as dst:
  53          dst.write(flood_prone[np.newaxis, :, :])
  54          dst.set_band_description(1, "FLOOD_PRONE")
  55      print(f"  Saved: {fp_path}")
  56  
  57      # --- Save exposure map ---
  58      exp_path = Path("data/processed") / f"{region}_exposure_map.tif"
  59      with rasterio.open(exp_path, "w", **out_profile) as dst:
  60          dst.write(exposure[np.newaxis, :, :])
  61          dst.set_band_description(1, "EXPOSED")
```

### Cross-region transfer evaluation: `src/builtup_mapping/train_rf_reverse.py` lines 1-40

```python
   1  """
   2  Reverse-direction transferability check: trains on Region B (Dakar) using
   3  a checkerboard spatial split, evaluates on B's own held-out tiles
   4  (same-region baseline) and on full Region A (Kelani) as the transfer test.
   5  Mirrors train_rf.py exactly, just with A and B swapped, so results are
   6  directly comparable.
   7  """
   8  import rasterio
   9  import numpy as np
  10  from sklearn.ensemble import RandomForestClassifier
  11  from sklearn.metrics import (
  12      accuracy_score, balanced_accuracy_score,
  13      precision_recall_fscore_support, f1_score, confusion_matrix,
  14  )
  15  from pathlib import Path
  16  import joblib
  17  
  18  FEATURE_BANDS = [
  19      "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
  20      "NDVI", "NDBI", "S1_VV", "S1_VH", "DEM",
  21  ]
  22  TILE_SIZE = 64
  23  
  24  
  25  def load_region(region):
  26      stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
  27      label_path = Path("data/processed") / f"{region}_builtup_label.tif"
  28      with rasterio.open(stack_path) as src:
  29          names = list(src.descriptions)
  30          bands = src.read()
  31          water_mask = bands[names.index("WATER_MASK")]
  32      with rasterio.open(label_path) as src:
  33          label = src.read(1)
  34      feature_idx = [names.index(b) for b in FEATURE_BANDS]
  35      return bands[feature_idx], label, water_mask
  36  
  37  
  38  def checkerboard_mask(H, W, tile_size):
  39      row_tile = (np.arange(H) // tile_size)[:, None]
  40      col_tile = (np.arange(W) // tile_size)[None, :]
```

### Cross-region transfer evaluation: `src/builtup_mapping/train_rf.py` lines 1-40

```python
   1  """
   2  Trains a Random Forest built-up classifier on Region A (Kelani) using a
   3  CHECKERBOARD spatial split (avoids the distribution-shift problem of a
   4  single left/right cut), then evaluates on:
   5    1. Held-out checkerboard tiles within Region A (same-region baseline)
   6    2. Full Region B (Dakar) -- cross-region transfer test
   7  
   8  Reports balanced accuracy and macro-F1 alongside raw precision/recall/F1,
   9  since raw F1 is misleading when built-up class balance differs sharply
  10  between the two regions (as it does here: ~34% in A's test set vs ~77%
  11  in B).
  12  """
  13  import rasterio
  14  import numpy as np
  15  from sklearn.ensemble import RandomForestClassifier
  16  from sklearn.metrics import (
  17      accuracy_score, balanced_accuracy_score,
  18      precision_recall_fscore_support, f1_score, confusion_matrix,
  19  )
  20  from pathlib import Path
  21  import joblib
  22  
  23  FEATURE_BANDS = [
  24      "B2", "B3", "B4", "B8", "B8A", "B11", "B12",
  25      "NDVI", "NDBI", "S1_VV", "S1_VH", "DEM",
  26  ]
  27  TILE_SIZE = 64  # pixels; checkerboard cell size
  28  
  29  
  30  def load_region(region):
  31      stack_path = Path("data/processed") / f"{region}_feature_stack.tif"
  32      label_path = Path("data/processed") / f"{region}_builtup_label.tif"
  33      with rasterio.open(stack_path) as src:
  34          names = list(src.descriptions)
  35          bands = src.read()
  36          water_mask = bands[names.index("WATER_MASK")]
  37      with rasterio.open(label_path) as src:
  38          label = src.read(1)
  39      feature_idx = [names.index(b) for b in FEATURE_BANDS]
  40      return bands[feature_idx], label, water_mask
```

# APPENDIX B MATERIAL (generated from the real outputs)

## B.1 Every output file (png, json, csv, md; GeoTIFF layers listed separately)

| File | Shows | Region | Size | Thesis status (from our chat; verify against your thesis) |
|---|---|---|---|---|
| `data/processed/dashboard_kpis.json` | Dashboard KPI values | both/NA | 2 KB | Source of Chapter 4 numbers |
| `data/processed/growth_projection_2030_2035.json` | 2030/2035 extrapolation | both/NA | 1 KB | Check in thesis; verify |
| `docs/phase5_transferability_synthesis.md` | Transfer performance | both/NA | 4 KB | Check in thesis; verify |
| `results/figures/region_a_kelani_builtup_classification_map.png` | Built-up classification map | A | 1731 KB | Used (Figures 4.2/4.3) |
| `results/figures/region_a_kelani_dem_void_mask.png` | Raw DEM void diagnostic | A | 251 KB | Check: likely Appendix E; verify |
| `results/figures/region_a_kelani_exposure_overlay.png` | Flood-exposure overlay | A | 2677 KB | Used (exposure figures), verify |
| `results/figures/region_a_kelani_growth_flood_overlay.png` | New growth 2000-2025 by flood-proneness (yellow/red) | A | 2736 KB | Used (Figure 4.6a/b) |
| `results/figures/region_a_kelani_hand_nan_mask.png` | HAND NaN diagnostic before buffered DEM | A | 332 KB | Used (Figures E.3/E.4) |
| `results/figures/region_a_kelani_s2_preview.png` | Sentinel-2 true-colour QA composite | A | 2148 KB | Check in thesis; verify |
| `results/figures/region_a_kelani_shap_exposed_vs_nonexposed.png` | SHAP, exposed vs non-exposed pixels | A | 53 KB | NOT confirmed in thesis: appendix candidate |
| `results/figures/region_a_kelani_shap_summary.png` | SHAP summary (surrogate) | A | 142 KB | Used (Figure 4.5), verify |
| `results/figures/region_b_dakar_builtup_classification_map.png` | Built-up classification map | B | 342 KB | Used (Figures 4.2/4.3) |
| `results/figures/region_b_dakar_dem_void_mask.png` | Raw DEM void diagnostic | B | 149 KB | Check: likely Appendix E; verify |
| `results/figures/region_b_dakar_exposure_overlay.png` | Flood-exposure overlay | B | 1562 KB | Used (exposure figures), verify |
| `results/figures/region_b_dakar_growth_flood_overlay.png` | New growth 2000-2025 by flood-proneness (yellow/red) | B | 1633 KB | Used (Figure 4.6a/b) |
| `results/figures/region_b_dakar_hand_nan_mask.png` | HAND NaN diagnostic before buffered DEM | B | 284 KB | Used (Figures E.3/E.4) |
| `results/figures/region_b_dakar_s2_preview.png` | Sentinel-2 true-colour QA composite | B | 1294 KB | Check in thesis; verify |
| `results/figures/region_b_dakar_shap_exposed_vs_nonexposed.png` | SHAP, exposed vs non-exposed pixels | B | 53 KB | NOT confirmed in thesis: appendix candidate |
| `results/figures/region_b_dakar_shap_summary.png` | SHAP summary (surrogate) | B | 166 KB | Used (Figure 4.5), verify |

GeoTIFF layers in data/processed:

| File | Size |
|---|---|
| `data/processed/region_a_kelani_builtup_2019.tif` | 0.4 MB |
| `data/processed/region_a_kelani_builtup_2019_norm.tif` | 0.5 MB |
| `data/processed/region_a_kelani_builtup_2021.tif` | 0.5 MB |
| `data/processed/region_a_kelani_builtup_2021_norm.tif` | 0.5 MB |
| `data/processed/region_a_kelani_builtup_2023.tif` | 0.4 MB |
| `data/processed/region_a_kelani_builtup_2023_norm.tif` | 0.4 MB |
| `data/processed/region_a_kelani_builtup_2025.tif` | 0.5 MB |
| `data/processed/region_a_kelani_builtup_2025_norm.tif` | 0.5 MB |
| `data/processed/region_a_kelani_builtup_label.tif` | 0.4 MB |
| `data/processed/region_a_kelani_exposure_map.tif` | 0.2 MB |
| `data/processed/region_a_kelani_feature_stack.tif` | 299.1 MB |
| `data/processed/region_a_kelani_flood_prone_mask.tif` | 0.2 MB |
| `data/processed/region_a_kelani_hand_10m.tif` | 24.0 MB |
| `data/processed/region_a_kelani_hand_native.tif` | 2.4 MB |
| `data/processed/region_b_dakar_builtup_2019.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_2019_norm.tif` | 0.4 MB |
| `data/processed/region_b_dakar_builtup_2021.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_2021_norm.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_2023.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_2023_norm.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_2025.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_2025_norm.tif` | 0.3 MB |
| `data/processed/region_b_dakar_builtup_label.tif` | 0.1 MB |
| `data/processed/region_b_dakar_exposure_map.tif` | 0.1 MB |
| `data/processed/region_b_dakar_feature_stack.tif` | 172.6 MB |
| `data/processed/region_b_dakar_flood_prone_mask.tif` | 0.1 MB |
| `data/processed/region_b_dakar_hand_10m.tif` | 13.3 MB |
| `data/processed/region_b_dakar_hand_native.tif` | 1.2 MB |

## B.2 Contents of every JSON and CSV output (as clean tables)

### `data/processed/dashboard_kpis.json`

| Field | region_a_kelani | region_b_dakar |
|---|---|---|
| estimated_exposed_buildings | 315864 | 330556 |
| exposed_builtup_km2 | 100.5882 | 95.2447 |
| ghsl_builtup_km2_by_epoch | [83.379076, 92.517185, 97.812022, 105.110206, 107.224277] | [49.819051, 52.621479, 54.493502, 57.545261, 57.812779] |
| ghsl_epochs | [2000, 2010, 2015, 2020, 2025] | [2000, 2010, 2015, 2020, 2025] |
| growth_p_value | 0.0007 | 0.0027 |
| growth_r_squared | 0.9861 | 0.9656 |
| growth_rate_km2_per_year | 1.0015 | 0.3453 |
| label | Lower Kelani Basin (Colombo/Gampaha, Sri Lanka) | Dakar Northern Suburbs (Senegal) |
| pct_buildings_exposed | 38.6479 | 71.4796 |
| pct_builtup_exposed | 39.1077 | 71.6934 |
| pct_land_flood_prone | 43.461 | 75.0528 |
| population_exposed | 650026.3125 | 1373273.25 |
| population_total_builtup | 1355485.875 | 1900385.5 |
| risk_pct_high | 20.4789 | 41.7807 |
| risk_pct_low | 60.8923 | 28.3066 |
| risk_pct_medium | 18.6288 | 29.9127 |
| total_buildings | 817287 | 462448 |
| total_builtup_km2 | 257.2082 | 132.85 |
| total_growth_pct_2000_2025 | 28.5985 | 16.0455 |

### `data/processed/growth_projection_2030_2035.json`

| Field | region_a_kelani | region_b_dakar |
|---|---|---|
| 2030.builtup_km2 | 113.2 | 60.0 |
| 2030.builtup_km2_95pi | [107.4, 119.0] | [56.8, 63.1] |
| 2030.exposed_builtup_km2 | 44.3 | 43.0 |
| 2030.exposed_builtup_km2_95pi | [42.0, 46.5] | [40.7, 45.3] |
| 2035.builtup_km2 | 118.2 | 61.7 |
| 2035.builtup_km2_95pi | [111.7, 124.8] | [58.1, 65.3] |
| 2035.exposed_builtup_km2 | 46.2 | 44.2 |
| 2035.exposed_builtup_km2_95pi | [43.7, 48.8] | [41.7, 46.8] |
