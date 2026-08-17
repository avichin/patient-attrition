"""
Survival-appropriate evaluation metrics. Deliberately NOT using accuracy/
F1/plain-AUC as the headline metrics for the survival models -- those are
classification metrics and don't account for censoring or the time
dimension. Use them only for the naive baseline classifier, and note in
your writeup that this is an apples-to-oranges comparison being made
explicit on purpose.

Metrics implemented:
- Harrell's C-index (concordance): already available via lifelines /
  scikit-survival `.score()`; wrapped here for a consistent interface.
- Time-dependent (cumulative/dynamic) AUC at fixed horizons: better than
  a single C-index for judging risk discrimination at a specific,
  clinically meaningful timepoint (e.g. "who will drop out in the next
  12 weeks").
- Calibration at fixed horizons: compares predicted vs. observed event
  probability within predicted-risk deciles, using Kaplan-Meier within
  each decile as the "observed" reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored


def time_dependent_auc(y_train_struct, y_test_struct, risk_scores: np.ndarray,
                         horizons_weeks: list[float]) -> pd.DataFrame:
    """
    risk_scores: higher = higher risk, aligned row-for-row with y_test_struct.
    y_*_struct: structured arrays from sksurv.util.Surv.from_dataframe.
    """
    aucs, mean_auc = cumulative_dynamic_auc(
        y_train_struct, y_test_struct, risk_scores, horizons_weeks
    )
    return pd.DataFrame({"horizon_weeks": horizons_weeks, "auc": aucs}), mean_auc


def concordance(y_struct, risk_scores: np.ndarray) -> float:
    event_field, time_field = y_struct.dtype.names
    c_index, *_ = concordance_index_censored(
        y_struct[event_field], y_struct[time_field], risk_scores
    )
    return c_index


def calibration_by_risk_decile(df: pd.DataFrame, risk_col: str, time_col: str,
                                  event_col: str, horizon_weeks: float,
                                  n_bins: int = 5) -> pd.DataFrame:
    """
    Bin patients into risk deciles (or n_bins groups), fit a KM curve within
    each bin, and report KM-estimated event probability at `horizon_weeks`
    as the "observed" rate to compare against mean predicted risk in that
    bin. This is a coarse but standard way to sanity-check calibration for
    survival models without needing a single scalar predicted probability
    per patient (which Cox models don't directly give you).
    """
    df = df.copy()
    df["risk_bin"] = pd.qcut(df[risk_col], n_bins, labels=False, duplicates="drop")

    rows = []
    for b, group in df.groupby("risk_bin"):
        kmf = KaplanMeierFitter()
        kmf.fit(group[time_col], event_observed=group[event_col])
        survival_at_horizon = kmf.predict(horizon_weeks)
        observed_event_prob = 1 - survival_at_horizon
        rows.append({
            "risk_bin": b,
            "mean_predicted_risk": group[risk_col].mean(),
            "n_patients": len(group),
            "observed_event_prob_at_horizon": observed_event_prob,
        })
    return pd.DataFrame(rows).sort_values("mean_predicted_risk")


if __name__ == "__main__":
    print("Import this module from a model script after fitting -- "
          "see models/survival_forest.py and models/cox_model.py for "
          "usage of concordance() and time_dependent_auc().")
