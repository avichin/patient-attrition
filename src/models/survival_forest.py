"""
Random Survival Forest via scikit-survival, for non-linear hazard
relationships the Cox model's linear-in-log-hazard assumption can't
capture (e.g. threshold effects in AE severity).

Uses the static (one-row-per-patient) view rather than the counting-process
format -- scikit-survival's RandomSurvivalForest expects a single time/event
per subject. If you want time-varying covariates in a forest, look at
extensions like landmarking or discrete-time survival forests as a
follow-up extension (documented as a TODO below, not implemented here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


FEATURE_COLUMNS = ["age", "baseline_severity", "comorbidity_count"]


def load_static_features(patients_path: str, visits_path: str) -> pd.DataFrame:
    patients = pd.read_csv(patients_path)
    visits = pd.read_csv(visits_path)

    # Summarize time-varying visit info into static per-patient features
    # (max AE severity observed, any lab abnormality, missed-visit rate) --
    # a reasonable static-model compromise, distinct from the full
    # time-varying treatment in cox_model.py.
    visit_summary = visits.groupby("patient_id").agg(
        max_ae_severity=("ae_severity_score", "max"),
        any_lab_abnormal=("lab_abnormal_flag", "max"),
        missed_visit_rate=("missed_visit_flag", "mean"),
    ).reset_index()

    df = patients.merge(visit_summary, on="patient_id", how="left")
    df["arm_treatment"] = (df["arm"] == "treatment").astype(int)
    return df


def fit_random_survival_forest(df: pd.DataFrame):
    features = FEATURE_COLUMNS + ["arm_treatment", "max_ae_severity",
                                    "any_lab_abnormal", "missed_visit_rate"]
    X = df[features].fillna(0)
    y = Surv.from_dataframe("event_observed", "observed_time_weeks", df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    rsf = RandomSurvivalForest(
        n_estimators=300, min_samples_split=10, min_samples_leaf=15,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )
    rsf.fit(X_train, y_train)
    c_index = rsf.score(X_test, y_test)
    return rsf, c_index, (X_test, y_test)


def main() -> None:
    df = load_static_features(
        "data/simulated_trial_patients.csv", "data/simulated_trial_visits.csv"
    )
    model, c_index, _ = fit_random_survival_forest(df)
    print(f"Random Survival Forest concordance index: {c_index:.3f}")

    # TODO: add permutation feature importance (sksurv doesn't give
    # split-based importances directly) and SHAP via `shap`'s KernelExplainer
    # or TreeExplainer where compatible.


if __name__ == "__main__":
    main()
