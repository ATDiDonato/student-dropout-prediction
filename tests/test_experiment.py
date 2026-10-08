from pathlib import Path

import pandas as pd

from src.experiment import (
    GROUP,
    STAGE_3_ASSESSMENT_FEATURES,
    STAGE_FEATURES,
    TARGET,
    load_stage_data,
    make_grouped_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_target_and_identifiers_are_not_features() -> None:
    for features in STAGE_FEATURES.values():
        assert TARGET not in features
        assert GROUP not in features
        assert "CompletedCourse" not in features


def test_stage_rows_are_aligned() -> None:
    datasets = load_stage_data(PROJECT_ROOT / "data" / "processed")
    reference = datasets["Stage 1"][[GROUP, TARGET]].reset_index(drop=True)
    for frame in datasets.values():
        pd.testing.assert_frame_equal(
            frame[[GROUP, TARGET]].reset_index(drop=True), reference
        )


def test_grouped_split_has_no_learner_overlap() -> None:
    datasets = load_stage_data(PROJECT_ROOT / "data" / "processed")
    reference = datasets["Stage 1"][[GROUP, TARGET]]
    split = make_grouped_split(reference)
    groups = {
        name: set(reference.loc[split.eq(name), GROUP])
        for name in ("train", "validation", "test")
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]


def test_stage_3_missing_assessment_rows_are_outcome_adjacent() -> None:
    datasets = load_stage_data(PROJECT_ROOT / "data" / "processed")
    stage_3 = datasets["Stage 3"]
    missing = stage_3[STAGE_3_ASSESSMENT_FEATURES].isna().all(axis=1)
    assert missing.sum() == 2231
    assert stage_3.loc[missing, TARGET].eq(1).all()

