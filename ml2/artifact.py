"""
Artifact management and serialization module for HVEAC Brain v1.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional, Tuple
import joblib

from ml.config import (
    PathConfig,
    TARGET_COLUMN,
    TARGET_CLASSES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    RANDOM_SEED,
)


@dataclass
class HVEACBrainArtifact:
    """Loaded model package container for inference."""
    model: Any
    preprocessor: Any
    schema: Dict[str, Any]
    metadata: Dict[str, Any]


def save_brain_artifacts(
    model: Any,
    preprocessor: Any,
    schema_path: Path,
    val_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]],
    model_name: str,
    hyperparameters: Dict[str, Any],
    output_dirs: Optional[list] = None,
) -> Dict[str, Path]:
    """
    Save complete, versioned ML artifacts according to Section 19 & 23.
    """
    if output_dirs is None:
        paths = PathConfig()
        output_dirs = [paths.models_dir, paths.artifacts_dir]

    # Read original schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Compile comprehensive model metadata
    model_metadata = {
        "artifact_version": "1.0.0",
        "brain_version": "hveac_brain_v1",
        "dataset_version": "1.1.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selected_model_architecture": model_name,
        "target_column": TARGET_COLUMN,
        "target_classes": TARGET_CLASSES,
        "class_to_index": CLASS_TO_INDEX,
        "index_to_class": {str(k): v for k, v in INDEX_TO_CLASS.items()},
        "feature_count": len(schema.get("feature_columns", [])),
        "encoded_feature_count": len(preprocessor.get_feature_names_out()) if getattr(preprocessor, "is_fitted", False) else 0,
        "hyperparameters": hyperparameters,
        "random_seed": RANDOM_SEED,
        "validation_metrics": val_metrics,
        "final_test_metrics": test_metrics,
        "status": "PRODUCTION_READY_OFFLINE_BENCHMARK",
    }

    saved_paths = {}

    for out_dir in output_dirs:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_file = out_dir / "hveac_brain_v1.joblib"
        prep_file = out_dir / "preprocessing_v1.joblib"
        schema_file = out_dir / "feature_schema_v1.json"
        meta_file = out_dir / "model_metadata.json"

        # Serialize model & preprocessor
        joblib.dump(model, model_file)
        joblib.dump(preprocessor, prep_file)

        # Save schema copy
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        # Save metadata
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(model_metadata, f, indent=2)

        saved_paths[str(out_dir)] = model_file

    return saved_paths


def load_brain_artifacts(artifact_dir: Path) -> HVEACBrainArtifact:
    """
    Load brain artifacts for inference and offline serving.
    """
    artifact_dir = Path(artifact_dir)
    model_file = artifact_dir / "hveac_brain_v1.joblib"
    prep_file = artifact_dir / "preprocessing_v1.joblib"
    schema_file = artifact_dir / "feature_schema_v1.json"
    meta_file = artifact_dir / "model_metadata.json"

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact missing: {model_file}")
    if not prep_file.exists():
        raise FileNotFoundError(f"Preprocessor artifact missing: {prep_file}")

    model = joblib.load(model_file)
    preprocessor = joblib.load(prep_file)

    schema = {}
    if schema_file.exists():
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)

    metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return HVEACBrainArtifact(
        model=model,
        preprocessor=preprocessor,
        schema=schema,
        metadata=metadata,
    )
