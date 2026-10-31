import pandas as pd

from home_credit.segmentation import evaluate_segments, fit_segment_definitions


def test_segment_boundaries_are_frozen_and_unknowns_are_explicit():
    development = pd.DataFrame(
        {
            "amount": range(100),
            "kind": ["cash", "card"] * 50,
            "WEEK_NUM": [1] * 100,
        }
    )
    definitions = fit_segment_definitions(
        development, {"amount_band": "amount", "credit_kind": "kind"}
    )
    amount = next(item for item in definitions if item["name"] == "amount_band")
    assert amount["edges"][1:-1] == [24.75, 49.5, 74.25]

    gold = pd.DataFrame(
        {
            "case_id": range(8),
            "amount": [None, 0, 20, 40, 60, 80, 120, 1000],
            "kind": [None, "cash", "card", "other", "cash", "card", "cash", "other"],
        }
    )
    predictions = pd.DataFrame(
        {
            "case_id": range(8),
            "WEEK_NUM": [9] * 8,
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "calibrated_score": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5],
            "mode": ["full"] * 8,
            "model_version": ["v1"] * 8,
        }
    )
    result = evaluate_segments(
        gold, predictions, definitions, min_samples=2, min_events=1, include_weekly=False
    )
    kind_groups = set(result.loc[result.segment == "credit_kind", "group"])
    assert {"missing", "unknown", "cash", "card"} <= kind_groups
    assert (result.status == "not_estimable_low_sample_or_event_count").any()
