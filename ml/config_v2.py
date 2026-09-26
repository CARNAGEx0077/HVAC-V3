"""
Configuration module for HVEAC Brain V2 ML Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class PathConfigV2:
    """Paths for V2 data, schemas, reports, and artifacts."""
    base_dir: Path = field(default_factory=lambda: Path("."))
    dataset_dir: Path = field(default_factory=lambda: Path("dataset_v2"))
    
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
    def raw_csv(self) -> Path:
        return self.dataset_dir / "raw" / "all_scenarios.csv"
        
    @property
    def schema_json(self) -> Path:
        return self.dataset_dir / "metadata" / "feature_schema.json"
        
    @property
    def metadata_json(self) -> Path:
        return self.dataset_dir / "metadata" / "dataset_metadata.json"
        
    @property
    def models_dir(self) -> Path:
        return Path("models/hveac_brain_v2")
        
    @property
    def v1_models_dir(self) -> Path:
        return Path("models/hveac_brain_v1")
        
    @property
    def reports_dir(self) -> Path:
        return Path("ml/reports")


# Authoritative Target Definition for Brain V2
TARGET_COLUMN: str = "optimal_room_setpoint_c"
TARGET_CLASSES: List[float] = [22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0]
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

# Categorical Feature Vocabularies
CATEGORICAL_FEATURES: List[str] = [
    "ac1_state",
    "ac2_state",
    "ac3_state",
    "ac4_state",
    "computer_1_workload",
    "computer_2_workload",
    "computer_3_workload",
    "computer_4_workload",
    "computer_5_workload",
    "computer_6_workload",
    "computer_7_workload",
    "computer_8_workload",
    "computer_9_workload",
    "computer_10_workload",
]

AC_STATE_VOCABULARY: List[str] = ["OFF", "ON"]
COMPUTER_WORKLOAD_VOCABULARY: List[str] = [
    "IDLE",
    "LIGHT",
    "GENERAL",
    "CPU_INTENSIVE",
    "GPU_INTENSIVE",
    "HEAVY",
]
