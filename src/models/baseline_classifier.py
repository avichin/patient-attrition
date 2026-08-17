"""
Naive logistic regression baseline: predicts binary dropout (yes/no) from
baseline covariates only.

THIS IS DELIBERATELY THE WRONG WAY TO DO IT and is included as a strawman,
not because it's a good model. Two specific failure modes to call out in
your writeup / notebook comparison:

1. Informative censoring bias: patients followed for only a few weeks
   before administrative study end are treated identically to patients
   followed for the full 52 weeks and who genuinely never dropped out.
   The binary label conflates "observed to complete" with "would have
   completed given more follow-up," which are not the same thing.
2. No use of the time dimension: a patient who drops out in week 2 and one
   who drops out in week 40 get the same label, even though they represent
   very different risk profiles and would call for different interventions.

Compare this model's discrimination (AUC) against the survival models'
C-index in evaluation.py, and show the naive model's predicted-risk
ranking disagrees most with the survival model for early-vs-late dropouts.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = ["arm", "age", "baseline_severity", "comorbidity_count"]
CATEGORICAL = ["arm"]
NUMERIC = ["age", "baseline_severity", "comorbidity_count"]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
    ], remainder="passthrough")
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df["dropped_out"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    return {"model": pipeline, "test_auc": auc, "X_test": X_test, "y_test": y_test}


def main() -> None:
    df = pd.read_csv("data/baseline_only_for_naive_classifier.csv")
    results = train_and_evaluate(df)
    print(f"[NAIVE BASELINE] Test AUC: {results['test_auc']:.3f}")
    print("Note: AUC alone does not capture WHEN dropout happens or WHY -- "
          "compare against C-index and cause-specific hazards in the "
          "survival models before drawing conclusions.")


if __name__ == "__main__":
    main()
