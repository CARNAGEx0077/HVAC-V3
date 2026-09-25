"""
Configuration module for HVEAC Brain v1 ML Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class PathConfig:
    """Paths for data, schemas, reports, and artifacts."""
    base_dir: Path = field(default_factory=lambda: Path("."))
    dataset_dir: Path = field(default_factory=lambda: Path("dataset_v1.1"))
    
    @property
    def train_csv(self) -> Path:
        return self.dataset_dir / "train" / "train.csv"
        
    @property
    def val_csv(self) -> Path:
        return self.dataset_dir / "validation" / "validation.csv"
        
    @property
    def test_csv(self) -> Path:
        return self.dataset_dir / "test" / "test.csv"
        
    @property
    def schema_json(self) -> Path:
        return self.dataset_dir / "metadata" / "feature_schema.json"
        
    @property
    def metadata_json(self) -> Path:
        return self.dataset_dir / "metadata" / "dataset_metadata.json"
        
    @property
    def artifacts_dir(self) -> Path:
        return Path("ml/artifacts")
        
    @property
    def models_dir(self) -> Path:
        return Path("models/hveac_brain_v1")
        
    @property
    def reports_dir(self) -> Path:
        return Path("ml/reports")
        
    @property
    def figures_dir(self) -> Path:
        return Path("ml/reports/figures")
        
    @property
    def experiments_dir(self) -> Path:
        return Path("ml/experiments")


# Authoritative Target Definition
TARGET_COLUMN: str = "optimal_room_setpoint_c"
TARGET_CLASSES: List[float] = [24.5, 25.0, 25.5, 26.0, 26.5]
CLASS_TO_INDEX: Dict[float, int] = {c: i for i, c in enumerate(TARGET_CLASSES)}
INDEX_TO_CLASS: Dict[int, float] = {i: c for i, c in enumerate(TARGET_CLASSES)}

# Anti-Leakage & Column Metadata
FORBIDDEN_INPUT_COLUMNS: List[str] = [
    "scenario_id",
    "scenario_family",
    "run_id",
    "random_seed",
    "timestamp",
    "simulation_time_seconds",
    "optimal_room_setpoint_c",
    "optimal_ac1_setpoint_c",
    "optimal_ac2_setpoint_c",
    "optimal_ac3_setpoint_c",
    "optimal_ac4_setpoint_c",
    "optimal_ac1_cooling_level",
    "optimal_ac2_cooling_level",
    "optimal_ac3_cooling_level",
    "optimal_ac4_cooling_level",
    "optimization_cost",
    "label_reason",
]

EXPECTED_FEATURE_COUNT: int = 80
EXPECTED_METADATA_COUNT: int = 6
EXPECTED_TARGET_COUNT: int = 11
EXPECTED_TOTAL_COLUMNS: int = 97

EXPECTED_TRAIN_ROWS: int = 50400
EXPECTED_VAL_ROWS: int = 10800
EXPECTED_TEST_ROWS: int = 10800

EXPECTED_TRAIN_SCENARIOS: int = 70
EXPECTED_VAL_SCENARIOS: int = 15
EXPECTED_TEST_SCENARIOS: int = 15

# Global Reproducibility Seed
RANDOM_SEED: int = 42


@dataclass
class ModelCandidatesConfig:
    """Hyperparameter configurations for candidate models."""
    random_seed: int = RANDOM_SEED
    
    # Model 1: Logistic Regression
    logistic_regression: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 1000,
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
        "class_weight": None,
    })
    
    # Model 2: Random Forest
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 150,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "class_weight": None,
    })
    
    # Model 3: HistGradientBoostingClassifier (fast, robust gradient boosting)
    hist_gradient_boosting: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 150,
        "max_depth": 8,
        "learning_rate": 0.1,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": RANDOM_SEED,
        "class_weight": None,
    })
