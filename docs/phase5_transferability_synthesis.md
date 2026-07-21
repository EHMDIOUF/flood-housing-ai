# Phase 5: Cross-Region Transferability Synthesis

## Research Question
To what extent does a built-up classifier trained in one geographic/climatic
context (Sri Lanka's Kelani River basin) transfer to a structurally different
context (Senegal's Dakar coastal corridor), and vice versa?

## Method
A Random Forest classifier was trained independently in each direction:
- **Model A→B**: trained on Region A (Kelani), evaluated on a held-out
  spatial checkerboard split of Region A (same-region baseline) and on
  the full Region B (cross-region transfer test).
- **Model B→A**: trained on Region B (Dakar), evaluated symmetrically.

A checkerboard spatial split (64x64 pixel tiles) was used rather than a
simple geographic cut, to avoid distribution-shift artifacts between
train/test sets within the same region. Balanced accuracy and macro-F1
were used as the primary metrics, since raw F1 was found to be misleading
when built-up class balance differs sharply between regions (Region A
~54% built-up vs Region B ~77% built-up in respective test sets).

## Results

| Direction | Same-region baseline (Bal. Acc / Macro-F1) | Cross-region transfer (Bal. Acc / Macro-F1) | Degradation |
|---|---|---|---|
| Train A -> Test A / B | 0.882 / 0.881 | 0.756 / 0.766 | -0.126 / -0.115 |
| Train B -> Test B / A | 0.906 / 0.901 | 0.786 / 0.774 | -0.120 / -0.127 |

The transferability cost is **remarkably symmetric**: approximately 12
percentage points of degradation in balanced accuracy occurs in both
directions, regardless of which region serves as the training source.
This suggests the transfer gap reflects a general challenge of applying
a fixed spectral/SAR classifier across distinct urban morphologies and
imaging conditions, rather than an artifact of one region being
inherently "easier" or serving as a better source domain.

Critically, both cross-region models still perform substantially better
than chance (balanced accuracy ~0.50 would indicate no skill), indicating
**partial, not complete, transferability** -- a nuanced and defensible
finding rather than either extreme.

## Explanatory Mechanism (from Phase 3 SHAP Analysis)
SHAP-based feature attribution revealed that the two region-specific
models rely on **different evidence** to reach similar predictions:
- The Region-A-trained model relies primarily on NDBI (mean |SHAP| =
  0.244), consistent with established built-up index theory.
- The Region-B-trained model relies primarily on raw visible-band
  brightness (B2, B3) and SAR backscatter (S1_VV, S1_VH), with NDBI
  relegated to near-lowest importance (0.024).

This divergence indicates that each model partially learned
region-specific spectral/radar signatures (e.g., local rooftop
materials, atmospheric conditions) rather than a single universal
built-up concept -- a plausible mechanistic explanation for the
observed ~12-point transfer gap in both directions.

A bootstrap significance test further confirmed that feature importance
shifts between flood-exposed and non-exposed built-up pixels, while
statistically significant (p<0.05 for most features, n=4000/group), are
small in relative magnitude (roughly 5-10% shifts), indicating the
model's decision logic is not fundamentally altered by flood exposure
status -- a useful sanity property distinguishing accuracy variation
(which does differ by region, per above) from decision-process
integrity (which appears stable).

## Conclusion
Built-up classification transfers **partially and symmetrically**
between these two study regions, with an approximately 12-point
balanced-accuracy cost attributable to genuine differences in the
spectral/radar signatures each model learns to associate with built-up
land cover, rather than to asymmetric data quality or an inherently
"harder" target region. This has a direct practical implication for
proposed multi-region flood-exposure monitoring systems: **region-
specific fine-tuning or recalibration is likely necessary** for
deployment-grade accuracy, even when a globally-trained base model
provides a reasonable starting point.
