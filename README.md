# Patient Dropout / Attrition Prediction — Survival Framing

Predicts when and why patients drop out of a clinical trial, rather than
just whether they do. Dropout is modeled as a time-to-event outcome with
competing risks (adverse-event withdrawal vs. lack-of-efficacy withdrawal vs.
administrative withdrawal), not as a binary classification target.

## Why this framing matters

A binary "dropout yes/no" classifier throws away the time dimension and is
biased under informative censoring (patients followed for different lengths
of time). This repo deliberately includes a naive classifier baseline
(`src/models/baseline_classifier.py`) alongside the survival models so the
degradation is visible and quantified, not just asserted.

## Structure

```
dropout-survival-prediction/
├── src/
│   ├── simulate_data.py        # synthetic longitudinal data, known ground-truth hazards
│   ├── data_prep.py            # raw visits -> (start, stop, event) counting-process format
│   ├── evaluation.py           # C-index, time-dependent AUC, calibration
│   ├── visualize.py            # KM curves, cumulative incidence functions
│   └── models/
│       ├── baseline_classifier.py   # naive logistic regression (strawman baseline)
│       ├── cox_model.py             # Cox PH with time-varying covariates
│       ├── survival_forest.py       # Random Survival Forest
│       └── competing_risks.py       # cause-specific Cox models
├── notebooks/                  # exploratory analysis, model comparison writeups
├── tests/
└── data/                       # generated/simulated data lands here (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.simulate_data          # writes data/simulated_trial.csv
python -m src.data_prep              # writes data/survival_format.csv
python -m src.models.cox_model
python -m src.models.survival_forest
python -m src.models.competing_risks
```

<!-- ## What to fill in / extend

- [ ] Swap simulated data for a real longitudinal cohort if available (see README notes on ADNI/SEER as options).
- [ ] Add sensitivity analysis for informative censoring (e.g., pattern-mixture model) — currently only flagged as a limitation.
- [ ] Add SHAP explanations for the Random Survival Forest.
- [ ] Write up model comparison (C-index, calibration) in `notebooks/`. -->

## Known limitations

1. **Informative censoring**: if sicker patients both drop out more AND have
   worse-quality late data, the independence assumption behind standard
   survival estimators is violated. Flagged here, not solved — a real
   sensitivity analysis is a good extension.
2. Simulated data uses a Weibull ground-truth hazard; real dropout hazards
   are unlikely to be this well-behaved. Treat absolute performance numbers
   as illustrative, not as a claim about real trials.
