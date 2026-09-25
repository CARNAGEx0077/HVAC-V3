"""
Data loading and anti-leakage verification module for HVEAC Dataset v1.1.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ml.config import (
    PathConfig,
    TARGET_COLUMN,
    TARGET_CLASSES,
    FORBIDDEN_INPUT_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_METADATA_COUNT,
    EXPECTED_TARGET_COUNT,
    EXPECTED_TOTAL_COLUMNS,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_VAL_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_SCENARIOS,
    EXPECTED_VAL_SCENARIOS,
    EXPECTED_TEST_SCENARIOS,
)


@dataclass
class DatasetSplits:
    """Container holding loaded and verified train/val/test splits."""
    X_train: pd.DataFrame
    y_train: pd.Series
    meta_train: pd.DataFrame
    
    X_val: pd.DataFrame
    y_val: pd.Series
    meta_val: pd.DataFrame
    
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_test: pd.DataFrame
    
    feature_names: List[str]
    categorical_features: List[str]
    numeric_features: List[str]
    target_name: str
    target_classes: List[float]


def verify_feature_schema(schema_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate the authoritative feature schema.
    Returns (feature_columns, metadata_columns, target_columns).
    Fails loudly if schema violates v1.1 specifications.
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Authoritative schema file missing: {schema_path}")
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        
    feat_cols = schema.get("feature_columns", [])
    meta_cols = schema.get("metadata_columns", [])
    target_cols = schema.get("target_columns", [])
    total_cols = schema.get("total_columns", 0)
    
    if len(feat_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature count violation: expected {EXPECTED_FEATURE_COUNT}, got {len(feat_cols)}"
        )
    if len(meta_cols) != EXPECTED_METADATA_COUNT:
        raise ValueError(
            f"Metadata count violation: expected {EXPECTED_METADATA_COUNT}, got {len(meta_cols)}"
        )
    if len(target_cols) != EXPECTED_TARGET_COUNT:
        raise ValueError(
            f"Target count violation: expected {EXPECTED_TARGET_COUNT}, got {len(target_cols)}"
        )
    if total_cols != EXPECTED_TOTAL_COLUMNS:
        raise ValueError(
            f"Total column count violation: expected {EXPECTED_TOTAL_COLUMNS}, got {total_cols}"
        )
        
    # Check for leakage in feature list
    for col in feat_cols:
        if col in meta_cols:
            raise ValueError(f"CRITICAL LEAKAGE: Metadata column '{col}' present in features!")
        if col in target_cols:
            raise ValueError(f"CRITICAL LEAKAGE: Target column '{col}' present in features!")
        for prefix in ["optimal_", "future_", "optimization_cost", "label_reason"]:
            if col.startswith(prefix):
                raise ValueError(f"CRITICAL LEAKAGE: Forbidden prefix in feature '{col}'!")
                
    if TARGET_COLUMN not in target_cols:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from schema targets!")
        
    return feat_cols, meta_cols, target_cols


def load_dataset_v1_1(paths: PathConfig = None) -> DatasetSplits:
    """
    Load Dataset v1.1 partitions, apply authoritative feature schema selection,
    and verify zero data leakage.
    """
    if paths is None:
        paths = PathConfig()
        
    # 1. Authoritative Schema Verification
    feat_cols, meta_cols, target_cols = verify_feature_schema(paths.schema_json)
    
    # 2. Check File Existence
    for p in [paths.train_csv, paths.val_csv, paths.test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Dataset partition missing: {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"Dataset partition empty: {p}")
            
    # 3. Read raw CSV files
    train_df = pd.read_csv(paths.train_csv)
    val_df = pd.read_csv(paths.val_csv)
    test_df = pd.read_csv(paths.test_csv)
    
    # 4. Verify Row Counts
    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Train row count mismatch: expected {EXPECTED_TRAIN_ROWS}, got {len(train_df)}")
    if len(val_df) != EXPECTED_VAL_ROWS:
        raise ValueError(f"Validation row count mismatch: expected {EXPECTED_VAL_ROWS}, got {len(val_df)}")
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Test row count mismatch: expected {EXPECTED_TEST_ROWS}, got {len(test_df)}")
        
    # 5. Verify Scenario Disjointness and Scenario Counts
    train_scenarios = set(train_df["scenario_id"].unique())
    val_scenarios = set(val_df["scenario_id"].unique())
    test_scenarios = set(test_df["scenario_id"].unique())
    
    if len(train_scenarios) != EXPECTED_TRAIN_SCENARIOS:
        raise ValueError(f"Train scenarios mismatch: expected {EXPECTED_TRAIN_SCENARIOS}, got {len(train_scenarios)}")
    if len(val_scenarios) != EXPECTED_VAL_SCENARIOS:
        raise ValueError(f"Validation scenarios mismatch: expected {EXPECTED_VAL_SCENARIOS}, got {len(val_scenarios)}")
    if len(test_scenarios) != EXPECTED_TEST_SCENARIOS:
        raise ValueError(f"Test scenarios mismatch: expected {EXPECTED_TEST_SCENARIOS}, got {len(test_scenarios)}")
        
    train_val_overlap = train_scenarios & val_scenarios
    train_test_overlap = train_scenarios & test_scenarios
    val_test_overlap = val_scenarios & test_scenarios
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE: Scenario overlap detected across splits! "
            f"Train/Val: {train_val_overlap}, Train/Test: {train_test_overlap}, Val/Test: {val_test_overlap}"
        )
        
    # 6. Extract ONLY 80 physical features for X
    # Strictly reject any metadata or target column
    for col in FORBIDDEN_INPUT_COLUMNS:
        if col in feat_cols:
            raise ValueError(f"Forbidden column '{col}' cannot be an input feature!")
            
    X_train = train_df[feat_cols].copy()
    X_val = val_df[feat_cols].copy()
    X_test = test_df[feat_cols].copy()
    
    # 7. Extract Target
    y_train = train_df[TARGET_COLUMN].copy()
    y_val = val_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()
    
    # Verify target values
    for split_name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        unique_targets = set(y.unique())
        invalid_targets = unique_targets - set(TARGET_CLASSES)
        if invalid_targets:
            raise ValueError(
                f"Invalid target classes in {split_name}: {invalid_targets}. "
                f"Expected subset of {TARGET_CLASSES}"
            )
            
    # 8. Separate Metadata (for downstream grouping/diagnostics only, NEVER input to model)
    meta_train = train_df[meta_cols].copy()
    meta_val = val_df[meta_cols].copy()
    meta_test = test_df[meta_cols].copy()
    
    # 9. Classify Feature Types (Categorical vs Numeric)
    categorical_features = []
    numeric_features = []
    
    for col in feat_cols:
        is_str = (
            pd.api.types.is_string_dtype(X_train[col])
            or X_train[col].dtype == object
            or str(X_train[col].dtype) == "str"
        )
        if is_str:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
            
    # Sanity check: 14 categorical and 66 numeric
    if len(categorical_features) != 14 or len(numeric_features) != 66:
        raise ValueError(
            f"Unexpected feature partition: {len(categorical_features)} categorical, "
            f"{len(numeric_features)} numeric. Expected 14 and 66."
        )
        
    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        meta_train=meta_train,
        X_val=X_val,
        y_val=y_val,
        meta_val=meta_val,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        feature_names=feat_cols,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        target_name=TARGET_COLUMN,
        target_classes=TARGET_CLASSES,
    )
