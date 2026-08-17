"""
Cause-specific Cox models for competing dropout risks: AE withdrawal,
lack-of-efficacy withdrawal, and administrative withdrawal.

Why competing risks, not one lumped "dropout" event: these causes have
different clinical implications and different actionable responses (an
AE-driven dropout risk should trigger a safety review; a lack-of-efficacy
signal is a different conversation). Modeling them as a single event throws
this away and can even bias cause-specific effect estimates because it
implicitly treats one cause as a competing risk / censoring event for
another without accounting for it.

This module fits separate cause-specific Cox models (patients who
experience a *different* cause are censored at their event time for the
cause being modeled -- the standard cause-specific hazard approach). A
natural extension is a Fine-Gray subdistribution hazard model for direct
cumulative incidence modeling; see the TODO at the bottom for how you'd
add that with `lifelines`' `CoxPHFitter` + manual subdistribution weighting
or the R `cmprsk`/`riskRegression` packages via `rpy2` if you want exact
Fine-Gray estimates.
"""
from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter

CAUSES = ["ae_withdrawal", "lack_of_efficacy", "administrative"]
FEATURE_COLUMNS = ["age", "baseline_severity", "comorbidity_count"]


def fit_cause_specific_models(df: pd.DataFrame) -> dict[str, CoxPHFitter]:
    df = df.copy()
    df["arm_treatment"] = (df["arm"] == "treatment").astype(int)

    models = {}
    for cause in CAUSES:
        event_col = f"event_{cause}"
        cols = ["observed_time_weeks", event_col, "arm_treatment"] + FEATURE_COLUMNS
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df[cols], duration_col="observed_time_weeks", event_col=event_col)
        models[cause] = cph
    return models


def summarize(models: dict[str, CoxPHFitter]) -> pd.DataFrame:
    rows = []
    for cause, model in models.items():
        for covariate, row in model.summary.iterrows():
            rows.append({
                "cause": cause,
                "covariate": covariate,
                "hazard_ratio": row["exp(coef)"],
                "p_value": row["p"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv("data/competing_risks_format.csv")
    models = fit_cause_specific_models(df)
    summary = summarize(models)
    print(summary.pivot(index="covariate", columns="cause", values="hazard_ratio").round(2))

    # TODO (Fine-Gray extension): implement subdistribution hazard modeling
    # to get cumulative incidence estimates that properly account for
    # competing events, rather than the cause-specific hazards above which
    # answer a subtly different question ("hazard among those still at
    # risk" vs. "actual probability of this cause occurring by time t").
    # Document this distinction explicitly in your writeup -- it's a common
    # point of confusion even among people using competing risks methods.


if __name__ == "__main__":
    main()
