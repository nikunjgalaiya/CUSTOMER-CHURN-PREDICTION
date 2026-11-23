# Customer Churn — Baseline and Signal Review

A reproducible evaluation of the repository's existing 100,000-row subscription dataset. The business question is whether the available variables support useful churn ranking before considering a retention pilot.

## Finding

The current logistic-regression specification **does not outperform a no-signal baseline**. Holdout ROC AUC is **0.4958**, versus **0.5000** for the prior baseline; the paired bootstrap 95% interval is **0.4882–0.5044**. This is evidence to investigate the data and problem definition, not to deploy the classifier or claim reduced churn.

## Reproduce

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate.py
```

`reports/metrics.json` records the source hash, split, feature list, training-only CV, holdout metrics and uncertainty. `reports/holdout_predictions.csv` uses anonymous source-row indices. `reports/tenure_cohorts.csv` provides descriptive rates, not causal effects.

## Evaluation design

- Fixed 80/20 stratified train/holdout split, seed 42.
- Logistic regression with imputation and scaling inside the pipeline; five-fold training-only CV.
- Probability-based ROC AUC and average precision; accuracy, precision, recall, F1 and explicitly named confusion-matrix cells at a fixed 0.5 threshold.
- Features: subscription length, monthly bill and usage. Names, identifiers, age, gender and location are excluded.
- 200 paired holdout bootstrap samples for an approximate ROC AUC interval. The interval does not cover dataset shift or every source of uncertainty.
- Four automated checks verify source contract, probability-based AUC, confusion labels and no-positive-prediction behaviour.

## Business interpretation

Do not use this result to target real customers. First verify data provenance, label definition, timing, cancellation reasons and whether outcome-relevant features are available before the decision. A future pilot would need a prospective evaluation and a costed intervention; prediction alone cannot establish retention uplift.

## Dataset and legacy material

The spreadsheet was already present in this repository. Its original provenance and reuse licence have not been established. No new licence or claim of real customer origin is made. Do not redistribute it outside the permissions applicable to the original dataset; use a verified licensed source for future extensions.

The original notebook and pickle are retained as historical work. The notebook contains older exploratory comparisons and machine-specific paths; it is not the validated evaluation entrypoint. The new workflow retrains from source and **does not load the legacy pickle**. Never load an untrusted pickle.

Results reproduced on 23 November 2026. No employer deployment, production readiness, or realised commercial improvement is claimed.
