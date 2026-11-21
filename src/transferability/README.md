# Cross-region transferability
Transfer experiments (train on Region A, test on Region B, and the reverse) are run by
`src/builtup_mapping/train_rf.py` and `src/builtup_mapping/train_rf_reverse.py`.
Model comparison across six classifiers: `src/builtup_mapping/compare_models.py`
(outputs in `results/tables/`).
