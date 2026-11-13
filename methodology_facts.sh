#!/bin/bash
out=methodology_facts.txt
: > $out
sec(){ echo -e "\n===== $1 =====" >> $out; }
sec "regions.py"; cat src/config/regions.py >> $out
for f in src/ingestion/export_to_drive.py src/ingestion/export_s2_with_swir.py src/ingestion/export_ghsl.py src/ingestion/export_dem_buffered.py src/ingestion/export_open_buildings.py; do
  sec "$f"
  grep -n -i "date\|cloud\|median\|reduce\|ImageCollection\|FeatureCollection\|scale\|crs\|select\|mask\|QA\|SCL\|confidence\|WorldCover\|WorldPop\|COPERNICUS\|GLO\|GHSL" $f >> $out
done
sec "generate_labels.py"; cat src/preprocessing/generate_labels.py >> $out
sec "build_feature_stack.py"; grep -n -i "resampl\|bilinear\|nearest\|ndvi\|ndbi\|water\|worldpop\|dem\|crs" src/preprocessing/build_feature_stack.py >> $out
sec "train_rf.py"; grep -n -i "checker\|tile\|64\|random_state\|class_weight\|n_estimators\|balanced\|print" src/builtup_mapping/train_rf.py >> $out
sec "build_exposure_map.py"; cat src/exposure/build_exposure_map.py >> $out
sec "compute_kpis_v2.py"; grep -n -i "pop\|sum\|tier" src/dashboard/compute_kpis_v2.py >> $out
wc -l $out
