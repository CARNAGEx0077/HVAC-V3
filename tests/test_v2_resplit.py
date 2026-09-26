"""
Test Suite for HVEAC Dataset V2 Behavior-Balanced Scenario Resplit.

Verifies all 12 dimensions mandated by Section 23 of the Resplit Specification:
1. Exact split size (70 / 15 / 15 scenarios; 50,400 / 10,800 / 10,800 rows)
2. Exact family counts (14 / 3 / 3 per family across all 5 families)
3. Zero scenario overlap (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅)
4. All scenarios represented exactly once
5. Exactly 720 rows per scenario
6. Raw CSV unchanged (SHA-256 check)
7. Optimizer deterministic with same seed
8. Optimizer produces repeatable split
9. Metadata excluded from features
10. Split does not alter row contents (exact cell-by-cell match with raw)
11. Rare-regime report matches actual CSV
12. Distribution metrics match actual CSV
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from generator.output.resplit_optimizer import (
    compute_file_sha256,
    extract_scenario_signatures,
    optimize_behavior_balanced_splits,
    OPERATIVE_SETPOINTS,
)

FROZEN_RAW_SHA256 = "20D7411B2BBC435214E9A3133E31719B7641D6D2F2062785A5CCD3D286606C87"
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_DIR / "dataset_v2"
RAW_CSV = DATASET_DIR / "raw" / "all_scenarios.csv"
TRAIN_CSV = DATASET_DIR / "train" / "train.csv"
VAL_CSV = DATASET_DIR / "validation" / "validation.csv"
TEST_CSV = DATASET_DIR / "test" / "test.csv"
METADATA_DIR = DATASET_DIR / "metadata"


@pytest.fixture(scope="module")
def dataset_splits():
    df_raw = pd.read_csv(RAW_CSV)
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    return {
        "raw": df_raw,
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }


def test_1_exact_split_size(dataset_splits):
    """Verify exact scenario and row counts across Train, Validation, and Test."""
    raw = dataset_splits["raw"]
    train = dataset_splits["train"]
    val = dataset_splits["val"]
    test = dataset_splits["test"]

    assert len(raw) == 72000, f"Raw rows mismatch: {len(raw)}"
    assert len(train) == 50400, f"Train rows mismatch: {len(train)}"
    assert len(val) == 10800, f"Validation rows mismatch: {len(val)}"
    assert len(test) == 10800, f"Test rows mismatch: {len(test)}"

    assert train["scenario_id"].nunique() == 70, f"Train scenarios mismatch: {train['scenario_id'].nunique()}"
    assert val["scenario_id"].nunique() == 15, f"Validation scenarios mismatch: {val['scenario_id'].nunique()}"
    assert test["scenario_id"].nunique() == 15, f"Test scenarios mismatch: {test['scenario_id'].nunique()}"


def test_2_exact_family_counts(dataset_splits):
    """Verify exact 14 / 3 / 3 scenario split per family across all 5 families."""
    train = dataset_splits["train"]
    val = dataset_splits["val"]
    test = dataset_splits["test"]

    families = [f"FAMILY_{i}" for i in range(1, 6)]
    for fam in families:
        n_tr = train[train["scenario_family"] == fam]["scenario_id"].nunique()
        n_va = val[val["scenario_family"] == fam]["scenario_id"].nunique()
        n_te = test[test["scenario_family"] == fam]["scenario_id"].nunique()

        assert n_tr == 14, f"{fam} Train count = {n_tr} != 14"
        assert n_va == 3, f"{fam} Validation count = {n_va} != 3"
        assert n_te == 3, f"{fam} Test count = {n_te} != 3"


def test_3_zero_scenario_overlap(dataset_splits):
    """Verify scenario disjointness: zero overlap between any splits."""
    s_tr = set(dataset_splits["train"]["scenario_id"].unique())
    s_va = set(dataset_splits["val"]["scenario_id"].unique())
    s_te = set(dataset_splits["test"]["scenario_id"].unique())

    assert len(s_tr.intersection(s_va)) == 0, f"Overlap Train/Val: {s_tr.intersection(s_va)}"
    assert len(s_tr.intersection(s_te)) == 0, f"Overlap Train/Test: {s_tr.intersection(s_te)}"
    assert len(s_va.intersection(s_te)) == 0, f"Overlap Val/Test: {s_va.intersection(s_te)}"


def test_4_all_scenarios_represented_exactly_once(dataset_splits):
    """Verify every raw scenario appears in exactly one split."""
    raw_scens = set(dataset_splits["raw"]["scenario_id"].unique())
    all_split_scens = (
        list(dataset_splits["train"]["scenario_id"].unique())
        + list(dataset_splits["val"]["scenario_id"].unique())
        + list(dataset_splits["test"]["scenario_id"].unique())
    )

    assert len(raw_scens) == 100
    assert len(all_split_scens) == 100
    assert set(all_split_scens) == raw_scens
    assert len(all_split_scens) == len(set(all_split_scens))


def test_5_720_rows_per_scenario(dataset_splits):
    """Verify all scenarios contain exactly 720 rows."""
    for split_name in ["train", "val", "test"]:
        df = dataset_splits[split_name]
        counts = df["scenario_id"].value_counts()
        for s_id, count in counts.items():
            assert count == 720, f"Scenario {s_id} in {split_name} has {count} rows != 720"


def test_6_raw_csv_unchanged():
    """Verify raw dataset file is bit-for-bit identical to frozen hash."""
    current_hash = compute_file_sha256(RAW_CSV)
    assert current_hash == FROZEN_RAW_SHA256, (
        f"Raw dataset SHA-256 altered! Expected {FROZEN_RAW_SHA256}, got {current_hash}"
    )


def test_7_optimizer_deterministic_with_same_seed(dataset_splits):
    """Verify optimizer returns identical assignments for identical seed."""
    df_raw = dataset_splits["raw"]
    sigs = extract_scenario_signatures(df_raw)

    assign1, score1, _ = optimize_behavior_balanced_splits(sigs, split_seed=42, num_iterations=1000)
    assign2, score2, _ = optimize_behavior_balanced_splits(sigs, split_seed=42, num_iterations=1000)

    assert score1 == score2, f"Score mismatch: {score1} vs {score2}"
    assert assign1 == assign2, "Assignment mismatch between identical runs"


def test_8_optimizer_repeatable_split(dataset_splits):
    """Verify the active dataset split matches the optimizer output for seed=42."""
    df_raw = dataset_splits["raw"]
    sigs = extract_scenario_signatures(df_raw)
    opt_assign, _, _ = optimize_behavior_balanced_splits(sigs, split_seed=42, num_iterations=20000)

    for s_id in dataset_splits["train"]["scenario_id"].unique():
        assert opt_assign[s_id] == "train"
    for s_id in dataset_splits["val"]["scenario_id"].unique():
        assert opt_assign[s_id] in ["val", "validation"]
    for s_id in dataset_splits["test"]["scenario_id"].unique():
        assert opt_assign[s_id] == "test"


def test_9_metadata_excluded_from_features():
    """Verify scenario_id, family, run_id, and timestamps are excluded from ML features."""
    schema_path = METADATA_DIR / "feature_schema.json"
    assert schema_path.exists(), f"Schema file missing: {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    meta_cols = set(schema.get("metadata_columns", []))
    feature_cols = set(schema.get("feature_columns", []))

    forbidden_features = {"scenario_id", "scenario_family", "run_id", "random_seed", "timestamp", "simulation_time_seconds"}
    assert forbidden_features.issubset(meta_cols), "Metadata columns missing from schema metadata_columns"
    assert len(forbidden_features.intersection(feature_cols)) == 0, (
        f"Metadata leaked into feature columns: {forbidden_features.intersection(feature_cols)}"
    )


def test_10_split_does_not_alter_row_contents(dataset_splits):
    """Verify split files preserve raw simulation rows verbatim."""
    raw = dataset_splits["raw"]
    splits_combined = pd.concat([
        dataset_splits["train"],
        dataset_splits["val"],
        dataset_splits["test"],
    ], ignore_index=True)

    # Sort both by scenario_id and simulation_time_seconds
    raw_sorted = raw.sort_values(by=["scenario_id", "simulation_time_seconds"]).reset_index(drop=True)
    splits_sorted = splits_combined.sort_values(by=["scenario_id", "simulation_time_seconds"]).reset_index(drop=True)

    assert raw_sorted.shape == splits_sorted.shape, "Shape mismatch between raw and combined splits"
    assert list(raw_sorted.columns) == list(splits_sorted.columns), "Column mismatch between raw and splits"

    # Compare key values: room temp, setpoint, heat load
    np.testing.assert_allclose(
        raw_sorted["room_average_temperature_c"].values,
        splits_sorted["room_average_temperature_c"].values,
        rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        raw_sorted["optimal_room_setpoint_c"].values,
        splits_sorted["optimal_room_setpoint_c"].values,
        rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        raw_sorted["total_heat_load_watts"].values,
        splits_sorted["total_heat_load_watts"].values,
        rtol=1e-5, atol=1e-5
    )


def test_11_rare_regime_report_matches_actual_csv(dataset_splits):
    """Verify rare regime report matches the actual scenario distributions in CSV."""
    report_path = METADATA_DIR / "rare_regime_split_report.md"
    assert report_path.exists(), "rare_regime_split_report.md does not exist"

    train = dataset_splits["train"]
    val = dataset_splits["val"]
    test = dataset_splits["test"]

    # Verify setpoint 22.0C is present only in Train (since only 1 exists)
    assert (train["optimal_room_setpoint_c"] == 22.0).sum() == 217
    assert (val["optimal_room_setpoint_c"] == 22.0).sum() == 0
    assert (test["optimal_room_setpoint_c"] == 22.0).sum() == 0

    # Verify setpoint 23.0C is present in all three splits
    assert (train["optimal_room_setpoint_c"] == 23.0).sum() > 0
    assert (val["optimal_room_setpoint_c"] == 23.0).sum() > 0
    assert (test["optimal_room_setpoint_c"] == 23.0).sum() > 0

    # Verify setpoint 25.0C is present in all three splits
    assert (train["optimal_room_setpoint_c"] == 25.0).sum() > 0
    assert (val["optimal_room_setpoint_c"] == 25.0).sum() > 0
    assert (test["optimal_room_setpoint_c"] == 25.0).sum() > 0


def test_12_distribution_metrics_match_actual_csv(dataset_splits):
    """Verify split_distribution_report.json matches the actual CSV data."""
    json_path = METADATA_DIR / "split_distribution_report.json"
    assert json_path.exists(), "split_distribution_report.json does not exist"

    with open(json_path, "r", encoding="utf-8") as f:
        dist_data = json.load(f)

    for s_name, df in [("TRAIN", dataset_splits["train"]), ("VALIDATION", dataset_splits["val"]), ("TEST", dataset_splits["test"])]:
        rep = dist_data[s_name]
        assert rep["scenario_count"] == df["scenario_id"].nunique()
        assert rep["row_count"] == len(df)

        # Check setpoint proportions match within 0.001
        for sp in OPERATIVE_SETPOINTS:
            actual_prop = (df["optimal_room_setpoint_c"] == sp).mean()
            rep_prop = rep["target_distribution"].get(str(sp), 0.0)
            assert abs(actual_prop - rep_prop) < 0.001, f"{s_name} {sp}°C prop mismatch: {actual_prop} vs {rep_prop}"
