"""
Heuristic baseline models for HVEAC setpoint prediction benchmarking.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ml.config import TARGET_CLASSES


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline A: Always predicts the most common class in the training partition.
    """

    def __init__(self):
        self.majority_class_: Optional[str] = None
        self.majority_setpoint_c_: Optional[float] = None
        self.classes_: np.ndarray = np.array([str(c) for c in TARGET_CLASSES])
        self.class_probabilities_: Optional[np.ndarray] = None

    def fit(self, X, y) -> "MajorityClassBaseline":
        y_str = pd.Series(y).astype(str)
        self.majority_class_ = y_str.mode()[0]
        self.majority_setpoint_c_ = float(self.majority_class_)
        
        # Calculate empirical class frequencies
        counts = y_str.value_counts(normalize=True)
        self.class_probabilities_ = np.array([counts.get(c, 0.0) for c in self.classes_])
        return self

    def predict(self, X) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.majority_class_)

    def predict_proba(self, X) -> np.ndarray:
        n_samples = len(X)
        return np.tile(self.class_probabilities_, (n_samples, 1))


class ConstantSetpointBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline B: Always predicts a constant 25.5°C setpoint.
    """

    def __init__(self, constant_setpoint: float = 25.5):
        self.constant_setpoint = constant_setpoint
        self.constant_class_ = str(constant_setpoint)
        self.classes_: np.ndarray = np.array([str(c) for c in TARGET_CLASSES])

    def fit(self, X, y=None) -> "ConstantSetpointBaseline":
        return self

    def predict(self, X) -> np.ndarray:
        n_samples = len(X)
        return np.full(n_samples, self.constant_class_)

    def predict_proba(self, X) -> np.ndarray:
        n_samples = len(X)
        proba = np.zeros(len(self.classes_))
        idx = list(self.classes_).index(self.constant_class_)
        proba[idx] = 1.0
        return np.tile(proba, (n_samples, 1))
