"""
Sanity tests. Not exhaustive -- extend with edge cases (zero-visit
patients, all-censored cohorts, single-arm data) as you build the project
out. Included to demonstrate the habit of testing data transformations,
not as complete coverage.
"""
import pandas as pd
import pytest

from src.simulate_data import simulate_patients, simulate_visit_level
from src.data_prep import (
    build_counting_process_format,
    build_baseline_only_table,
    build_competing_risks_table,
)


@pytest.fixture(scope="module")
def small_cohort():
    patients = simulate_patients(n_patients=50, seed=1)
    visits = simulate_visit_level(patients, seed=1)
    return patients, visits


def test_counting_process_events_sum_correctly(small_cohort):
    patients, visits = small_cohort
    cp = build_counting_process_format(patients, visits)
    n_events_in_cp = cp["event"].sum()
    n_events_in_patients = patients["event_observed"].sum()
    assert n_events_in_cp == n_events_in_patients


def test_counting_process_intervals_are_monotonic(small_cohort):
    patients, visits = small_cohort
    cp = build_counting_process_format(patients, visits)
    assert (cp["stop"] > cp["start"]).all()


def test_baseline_table_dropout_matches_event_cause(small_cohort):
    patients, _ = small_cohort
    baseline = build_baseline_only_table(patients)
    expected = patients["event_observed"] & (patients["event_cause"] != "completed")
    assert (baseline["dropped_out"].values == expected.values).all()


def test_competing_risks_indicators_are_mutually_exclusive(small_cohort):
    patients, _ = small_cohort
    cr = build_competing_risks_table(patients)
    cause_cols = ["event_ae_withdrawal", "event_lack_of_efficacy", "event_administrative"]
    assert (cr[cause_cols].sum(axis=1) <= 1).all()
