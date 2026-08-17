"""
Visualization helpers: Kaplan-Meier curves by arm/covariate, and cumulative
incidence functions (CIF) for competing risks -- CIFs, not separate KM
curves per cause, because KM applied naively to a single cause while
treating other-cause events as censoring overestimates that cause's true
incidence. This distinction is worth a sentence in your README; it's a
common and consequential mistake.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def plot_km_by_arm(df: pd.DataFrame, time_col: str = "observed_time_weeks",
                     event_col: str = "event_observed", arm_col: str = "arm",
                     ax=None):
    ax = ax or plt.gca()
    for arm, group in df.groupby(arm_col):
        kmf = KaplanMeierFitter(label=arm)
        kmf.fit(group[time_col], event_observed=group[event_col])
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Weeks")
    ax.set_ylabel("Survival probability (retained in study)")
    ax.set_title("Kaplan-Meier retention curves by arm")
    return ax


def plot_cumulative_incidence(df: pd.DataFrame, cause_cols: list[str],
                                 time_col: str = "observed_time_weeks", ax=None):
    """
    Naive Aalen-Johansen-style CIF approximation via cumulative sum of
    cause-specific hazard contributions. For a production-quality CIF,
    use `lifelines.AalenJohansenFitter` directly -- included here as a
    thin wrapper to keep the dependency surface visible.
    """
    from lifelines import AalenJohansenFitter

    ax = ax or plt.gca()
    # Build a single "any event" time/cause pair for AalenJohansenFitter.
    combined_cause = pd.Series("censored", index=df.index)
    for cause_col in cause_cols:
        combined_cause[df[cause_col] == 1] = cause_col

    for cause_col in cause_cols:
        ajf = AalenJohansenFitter()
        event_indicator = (combined_cause == cause_col).astype(int)
        ajf.fit(df[time_col], combined_cause.replace(
            {c: i + 1 for i, c in enumerate(cause_cols)}
        ).where(combined_cause != "censored", 0),
            event_of_interest=cause_cols.index(cause_col) + 1)
        ajf.plot(ax=ax, label=cause_col)

    ax.set_xlabel("Weeks")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title("Cumulative incidence by dropout cause")
    return ax


if __name__ == "__main__":
    patients = pd.read_csv("data/simulated_trial_patients.csv")
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    plot_km_by_arm(patients, ax=axes)
    plt.tight_layout()
    plt.savefig("data/km_by_arm.png", dpi=150)
    print("Saved data/km_by_arm.png")
