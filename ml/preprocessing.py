"""
Preprocessing pipeline for HVEAC Brain v1.

Strict Anti-Leakage Rule:
- Preprocessor must be fitted ONLY on training data.
- Validation and Test splits are transformed using the fitted preprocessor.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.config import (
    EXPECTED_FEATURE_COUNT,
    FORBIDDEN_INPUT_COLUMNS,
)


class HVEACPreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust feature preprocessor for HVEAC tabular physical state features.
    
    Encodes:
    - 66 continuous physical and lagged features using StandardScaler.
    - 14 categorical features (4 AC states, 10 computer workloads) using OneHotEncoder.
    
    Ensures deterministic category alignment across all scenarios and splits.
    """

    # Domain-defined category vocabularies
    AC_STATES = ["OFF", "ON"]
    COMPUTER_WORKLOADS = [
        "IDLE",
        "LIGHT",
        "GENERAL",
        "CPU_INTENSIVE",
        "GPU_INTENSIVE",
        "HEAVY",
    ]

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        scale_numeric: bool = True,
    ):
        self.feature_names = feature_names
        self.scale_numeric = scale_numeric
        self.is_fitted = False
        
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.encoded_feature_names_: List[str] = []
        self.diagnostics_: Dict[str, Any] = {}

    def _inspect_and_validate_columns(self, X: pd.DataFrame):
        """Perform pre-training runtime sanity checks on raw DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X)}")
            
        cols = list(X.columns)
        
        # Verify no forbidden or metadata leakage columns
        for c in FORBIDDEN_INPUT_COLUMNS:
            if c in cols:
                raise ValueError(f"CRITICAL LEAKAGE: Forbidden column '{c}' passed to preprocessor!")

        if self.feature_names is not None:
            missing = [c for c in self.feature_names if c not in cols]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing[:5]} (total {len(missing)})")
            unexpected = [c for c in cols if c not in self.feature_names]
            if unexpected:
                raise ValueError(f"Unexpected columns found in preprocessor input: {unexpected[:5]}")
        else:
            if len(cols) != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"Expected exactly {EXPECTED_FEATURE_COUNT} input features, got {len(cols)}"
                )

        # Null and NaN checks
        null_counts = X.isnull().sum()
        if null_counts.sum() > 0:
            bad_cols = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Null / NaN values detected in features: {bad_cols}")

    def fit(self, X: pd.DataFrame, y=None) -> "HVEACPreprocessor":
        """
        Fit the preprocessor strictly on training data.
        """
        self._inspect_and_validate_columns(X)
        
        # Determine feature names and column partitioning
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        self.categorical_features_ = [
            c for c in self.feature_names
            if pd.api.types.is_string_dtype(X[c])
            or X[c].dtype == object
            or str(X[c].dtype) == "str"
        ]
        self.numeric_features_ = [
            c for c in self.feature_names if c not in self.categorical_features_
        ]

        # Verify exact counts
        if len(self.categorical_features_) != 14:
            raise ValueError(
                f"Expected 14 categorical features, found {len(self.categorical_features_)}: {self.categorical_features_}"
            )
        if len(self.numeric_features_) != 66:
            raise ValueError(
                f"Expected 66 numeric features, found {len(self.numeric_features_)}"
            )

        # Check for infinities in numeric columns
        num_vals = X[self.numeric_features_].values
        if np.isinf(num_vals).any():
            raise ValueError("Infinite values detected in numeric training columns!")

        # Diagnostics: check constant columns
        std_devs = X[self.numeric_features_].std()
        const_cols = std_devs[std_devs == 0].index.tolist()
        self.diagnostics_ = {
            "total_input_features": len(self.feature_names),
            "numeric_feature_count": len(self.numeric_features_),
            "categorical_feature_count": len(self.categorical_features_),
            "constant_numeric_columns": const_cols,
            "fitted_rows": len(X),
        }

        # Setup explicit category lists for deterministic one-hot alignment
        categories = []
        for col in self.categorical_features_:
            if "ac" in col and "state" in col:
                categories.append(self.AC_STATES)
            elif "workload" in col:
                categories.append(self.COMPUTER_WORKLOADS)
            else:
                categories.append(sorted(X[col].unique().tolist()))

        cat_transformer = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
        )
        
        num_transformer = StandardScaler() if self.scale_numeric else "passthrough"

        self.column_transformer_ = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.numeric_features_),
                ("cat", cat_transformer, self.categorical_features_),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

        self.column_transformer_.fit(X)
        self.encoded_feature_names_ = list(self.column_transformer_.get_feature_names_out())
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using the pre-fitted parameters.
        Can be applied to Validation, Test, or Inference data.
        """
        if not self.is_fitted or self.column_transformer_ is None:
            raise RuntimeError("HVEACPreprocessor must be fitted before calling transform!")
            
        self._inspect_and_validate_columns(X)
        
        # Check for infinite values in numeric columns
        num_vals = X[self.numeric_features_].values
        if np.isinf(num_vals).any():
            raise ValueError("Infinite values detected in numeric input columns during transform!")

        return self.column_transformer_.transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return names of all encoded output features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted yet.")
        return self.encoded_feature_names_
