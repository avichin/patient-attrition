"""
Simulate longitudinal clinical trial data with a KNOWN ground-truth hazard
function and competing dropout causes.

Why simulate at all: it lets us validate every downstream model against a
target we actually know, instead of only eyeballing plausibility on real
data where the true hazard is unobservable. This is the single most
convincing thing you can do in a portfolio project — see README.

Ground truth
------------
Each patient has a latent risk score built from covariates. Event time is
drawn from a Weibull distribution whose scale depends on that risk score
(proportional hazards by construction), so a correctly specified Cox model
should recover the true covariate effects asymptotically. Competing event
*types* (AE withdrawal, lack-of-efficacy withdrawal, administrative
withdrawal, and study completion / right-censoring) are assigned via a
multinomial draw conditioned on the covariates, so cause-specific hazards
genuinely differ across event types -- this is what makes the competing
risks model (src/models/competing_risks.py) meaningfully different from
a single-event Cox model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42
N_PATIENTS = 1500
MAX_FOLLOWUP_WEEKS = 52
VISIT_INTERVAL_WEEKS = 4


def _weibull_event_time(risk_score: np.ndarray, rng: np.random.Generator,
                          shape: float = 1.5, base_scale: float = 40.0) -> np.ndarray:
    """Higher risk_score -> smaller scale -> earlier events (PH-consistent)."""
    scale = base_scale * np.exp(-risk_score)
    u = rng.uniform(0, 1, size=len(risk_score))
    return scale * (-np.log(u)) ** (1 / shape)


def simulate_patients(n_patients: int = N_PATIENTS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(58, 12, n_patients).clip(18, 90)
    baseline_severity = rng.uniform(0, 10, n_patients)  # e.g. disease severity score
    arm = rng.choice(["treatment", "placebo"], n_patients, p=[0.5, 0.5])
    comorbidity_count = rng.poisson(1.2, n_patients)
    site_id = rng.integers(1, 21, n_patients)  # 20 simulated sites

    # Latent risk score drives time-to-event (ground truth for validation).
    treatment_effect = np.where(arm == "treatment", -0.4, 0.0)  # treatment reduces dropout risk
    risk_score = (
        0.03 * (age - 58)
        + 0.15 * (baseline_severity - 5)
        + 0.10 * comorbidity_count
        + treatment_effect
        + rng.normal(0, 0.3, n_patients)  # unobserved heterogeneity
    )

    event_time = _weibull_event_time(risk_score, rng)
    admin_censor_time = np.full(n_patients, MAX_FOLLOWUP_WEEKS, dtype=float)

    observed_time = np.minimum(event_time, admin_censor_time)
    had_event = event_time <= admin_censor_time

    # Cause of event, conditioned loosely on covariates so cause-specific
    # hazards genuinely differ (e.g. higher severity -> more AE-related
    # withdrawal; higher age -> more administrative withdrawal).
    cause = np.empty(n_patients, dtype=object)
    for i in range(n_patients):
        if not had_event[i]:
            cause[i] = "completed"
            continue
        p_ae = 0.15 + 0.03 * baseline_severity[i] / 10
        p_admin = 0.10 + 0.01 * (age[i] - 58) / 30
        p_admin = max(min(p_admin, 0.4), 0.05)
        p_ae = max(min(p_ae, 0.6), 0.05)
        p_efficacy = max(1 - p_ae - p_admin, 0.05)
        probs = np.array([p_ae, p_efficacy, p_admin])
        probs = probs / probs.sum()
        cause[i] = rng.choice(["ae_withdrawal", "lack_of_efficacy", "administrative"], p=probs)

    df = pd.DataFrame({
        "patient_id": [f"PT-{i:05d}" for i in range(n_patients)],
        "site_id": site_id,
        "arm": arm,
        "age": age.round(1),
        "baseline_severity": baseline_severity.round(2),
        "comorbidity_count": comorbidity_count,
        "true_risk_score": risk_score.round(3),  # kept for ground-truth validation only
        "observed_time_weeks": observed_time.round(1),
        "event_observed": had_event,
        "event_cause": cause,
    })
    return df


def simulate_visit_level(patient_df: pd.DataFrame, seed: int = RNG_SEED) -> pd.DataFrame:
    """
    Expand each patient into periodic visit records with time-varying
    covariates (AE severity, lab abnormality flag, visit compliance),
    truncated at the patient's observed event/censoring time. This is the
    "raw-ish" longitudinal format that data_prep.py turns into
    counting-process (start, stop, event) format for time-varying Cox
    models.
    """
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _, row in patient_df.iterrows():
        n_visits = max(1, int(row["observed_time_weeks"] // VISIT_INTERVAL_WEEKS))
        ae_severity_trend = rng.normal(0, 1)  # patient-level drift
        for v in range(n_visits):
            visit_week = (v + 1) * VISIT_INTERVAL_WEEKS
            ae_severity = max(0, rng.normal(1 + 0.05 * v * (ae_severity_trend > 0), 1))
            lab_abnormal = rng.uniform(0, 1) < (0.1 + 0.01 * ae_severity)
            missed_visit = rng.uniform(0, 1) < 0.05
            rows.append({
                "patient_id": row["patient_id"],
                "visit_week": visit_week,
                "ae_severity_score": round(ae_severity, 2),
                "lab_abnormal_flag": bool(lab_abnormal),
                "missed_visit_flag": bool(missed_visit),
            })
    return pd.DataFrame(rows)


def main() -> None:
    patients = simulate_patients()
    visits = simulate_visit_level(patients)

    patients.to_csv("data/simulated_trial_patients.csv", index=False)
    visits.to_csv("data/simulated_trial_visits.csv", index=False)

    print(f"Simulated {len(patients)} patients, {len(visits)} visit records.")
    print(patients["event_cause"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
