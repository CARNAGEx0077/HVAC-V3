"""
Preprocessing pipeline for HVEAC Brain V2.

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

from ml.config_v2 import (
    EXPECTED_FEATURE_COUNT,
    FORBIDDEN_INPUT_COLUMNS,
    CATEGORICAL_FEATURES,
    AC_STATE_VOCABULARY,
    COMPUTER_WORKLOAD_VOCABULARY,
)


class HVEACPreprocessorV2(BaseEstimator, TransformerMixin):
    """
    Robust feature preprocessor for HVEAC V2 tabular physical state features.
    
    Encodes:
    - 66 continuous physical and rolling-average features using StandardScaler.
    - 14 categorical features (4 AC states, 10 computer workloads) using OneHotEncoder.
    
    Guarantees:
    - Strict fit on TRAIN only.
    - Deterministic category alignment across splits and inference calls.
    - Explicit rejection of metadata columns, forbidden target leaks, NaNs, and unexpected keys.
    """

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

    def fit(self, X: pd.DataFrame, y=None) -> "HVEACPreprocessorV2":
        """
        Fit the preprocessor strictly on training data.
        """
        self._inspect_and_validate_columns(X)
        
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        self.categorical_features_ = [c for c in CATEGORICAL_FEATURES if c in self.feature_names]
        self.numeric_features_ = [c for c in self.feature_names if c not in self.categorical_features_]

        # Explicit known category definitions to prevent unseen category encoding bugs
        categories = []
        for cat_col in self.categorical_features_:
            if "ac" in cat_col and "state" in cat_col:
                categories.append(AC_STATE_VOCABULARY)
            elif "workload" in cat_col:
                categories.append(COMPUTER_WORKLOAD_VOCABULARY)
            else:
                categories.append("auto")

        transformers = []
        if self.numeric_features_:
            num_transformer = StandardScaler() if self.scale_numeric else "passthrough"
            transformers.append(("num", num_transformer, self.numeric_features_))

        if self.categorical_features_:
            cat_transformer = OneHotEncoder(
                categories=categories,
                handle_unknown="ignore",
                sparse_output=False,
            )
            transformers.append(("cat", cat_transformer, self.categorical_features_))

        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.column_transformer_.fit(X)

        # Store encoded feature names
        encoded_names = []
        if self.numeric_features_:
            encoded_names.extend(self.numeric_features_)
            
        if self.categorical_features_:
            cat_encoder = self.column_transformer_.named_transformers_["cat"]
            cat_names = cat_encoder.get_feature_names_out(self.categorical_features_)
            encoded_names.extend(list(cat_names))

        self.encoded_feature_names_ = encoded_names
        self.is_fitted = True
        
        self.diagnostics_ = {
            "input_feature_count": len(self.feature_names),
            "numeric_feature_count": len(self.numeric_features_),
            "categorical_feature_count": len(self.categorical_features_),
            "encoded_feature_count": len(self.encoded_feature_names_),
        }
        
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform raw feature DataFrame into scaled, encoded numerical array.
        """
        if not self.is_fitted or self.column_transformer_ is None:
            raise RuntimeError("Preprocessor must be fitted on training data before transform()")

        self._inspect_and_validate_columns(X)
        
        # Ensure column ordering matches fitted feature_names
        X_ordered = X[self.feature_names]
        
        transformed = self.column_transformer_.transform(X_ordered)
        
        # Sanity check on output array
        if np.isnan(transformed).any() or np.isinf(transformed).any():
            raise ValueError("Transformed feature matrix contains NaNs or Infs!")

        return transformed

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return names of output encoded features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to retrieve output feature names")
        return list(self.encoded_feature_names_)
