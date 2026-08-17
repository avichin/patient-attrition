"""
Transform raw patient + visit-level data into the formats each model needs:

1. Counting-process (start, stop, event) format for time-varying Cox models
   (lifelines' CoxTimeVaryingFitter expects one row per interval).
2. A static baseline-only table for the naive classifier baseline, so the
   comparison between "naive baseline features" and "full time-varying
   model" is apples-to-apples on the same underlying patients.
3. Cause-specific event indicator columns for competing risks models.
"""
from __future__ import annotations

import pandas as pd


def build_counting_process_format(patients: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (patient, interval). `event` is 1 only on the final interval
    for patients who had an event (not censored). Time-varying covariates
    (ae_severity_score, lab_abnormal_flag) come from the visit at the start
    of each interval.
    """
    visits = visits.sort_values(["patient_id", "visit_week"]).copy()
    records = []

    for pid, group in visits.groupby("patient_id"):
        patient_row = patients.loc[patients["patient_id"] == pid].iloc[0]
        group = group.reset_index(drop=True)
        prev_week = 0.0
        for i, visit in group.iterrows():
            start = prev_week
            stop = visit["visit_week"]
            is_last_interval = i == len(group) - 1
            event_here = bool(is_last_interval and patient_row["event_observed"])
            records.append({
                "patient_id": pid,
                "start": start,
                "stop": stop,
                "event": event_here,
                "event_cause": patient_row["event_cause"] if event_here else "censored",
                "arm": patient_row["arm"],
                "age": patient_row["age"],
                "baseline_severity": patient_row["baseline_severity"],
                "comorbidity_count": patient_row["comorbidity_count"],
                "ae_severity_score": visit["ae_severity_score"],
                "lab_abnormal_flag": int(visit["lab_abnormal_flag"]),
                "missed_visit_flag": int(visit["missed_visit_flag"]),
            })
            prev_week = stop

    return pd.DataFrame(records)


def build_baseline_only_table(patients: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline-only covariates + binary dropout label, used ONLY by the naive
    classifier baseline in models/baseline_classifier.py. Deliberately
    excludes time-varying information and the actual observed_time_weeks,
    to mirror how a naive "predict dropout at baseline" model would
    actually be built in practice (and to make its blind spots visible).
    """
    df = patients.copy()
    df["dropped_out"] = df["event_observed"] & (df["event_cause"] != "completed")
    return df[[
        "patient_id", "arm", "age", "baseline_severity",
        "comorbidity_count", "dropped_out",
    ]]


def build_competing_risks_table(patients: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with a cause-specific event indicator per cause."""
    df = patients.copy()
    for cause in ["ae_withdrawal", "lack_of_efficacy", "administrative"]:
        df[f"event_{cause}"] = (df["event_cause"] == cause).astype(int)
    return df


def main() -> None:
    patients = pd.read_csv("data/simulated_trial_patients.csv")
    visits = pd.read_csv("data/simulated_trial_visits.csv")

    counting_process = build_counting_process_format(patients, visits)
    baseline_only = build_baseline_only_table(patients)
    competing_risks = build_competing_risks_table(patients)

    counting_process.to_csv("data/survival_format_counting_process.csv", index=False)
    baseline_only.to_csv("data/baseline_only_for_naive_classifier.csv", index=False)
    competing_risks.to_csv("data/competing_risks_format.csv", index=False)

    print(f"Counting-process rows: {len(counting_process)}")
    print(f"Baseline-only rows: {len(baseline_only)}")
    print(f"Competing-risks rows: {len(competing_risks)}")


if __name__ == "__main__":
    main()
