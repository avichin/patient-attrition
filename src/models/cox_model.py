"""
Cox proportional hazards model with TIME-VARYING covariates, using
lifelines' CoxTimeVaryingFitter on the counting-process (start, stop,
event) format produced by data_prep.py.

Why time-varying and not a static Cox model: AE severity and lab
abnormalities change over the course of follow-up and are exactly the
kind of signal a site coordinator would want reflected in an
up-to-date risk estimate, not frozen at baseline.
"""
from __future__ import annotations

import pandas as pd
from lifelines import CoxTimeVaryingFitter


FEATURE_COLUMNS = [
    "age", "baseline_severity", "comorbidity_count",
    "ae_severity_score", "lab_abnormal_flag", "missed_visit_flag",
]


def prepare_for_lifelines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["arm_treatment"] = (out["arm"] == "treatment").astype(int)
    return out[["patient_id", "start", "stop", "event", "arm_treatment"] + FEATURE_COLUMNS]


def fit_cox_time_varying(df: pd.DataFrame) -> CoxTimeVaryingFitter:
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(
        df,
        id_col="patient_id",
        start_col="start",
        stop_col="stop",
        event_col="event",
        show_progress=False,
    )
    return ctv


def main() -> None:
    raw = pd.read_csv("data/survival_format_counting_process.csv")
    prepped = prepare_for_lifelines(raw)

    model = fit_cox_time_varying(prepped)
    print(model.summary[["coef", "exp(coef)", "p"]])

    # TODO: compare fitted coefficients against the ground-truth risk_score
    # weights baked into simulate_data.py to sanity-check model recovery.
    # TODO: save model.summary to a results/ dir for the model-comparison
    # writeup in notebooks/.


if __name__ == "__main__":
    main()
