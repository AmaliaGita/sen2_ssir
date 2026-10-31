"""Public API for the Sen2-SSIR implementation."""

from .config import (
    EVALUATION_RESULTS_DIR,
    INFERENCE_RESULTS_DIR,
    PREPARED_DATA_DIR,
    PRETRAINED_WEIGHTS,
    TESTING_SCENE_DIR,
    TRAINING_RESULTS_DIR,
    TRAINING_SCENE_DIR,
    InferenceConfig,
    PatchConfig,
    TrainingConfig,
    ensure_project_directories,
)
from .evaluation import evaluate_scene
from .inference import predict_tiled_to_npy, superresolve_scene_to_geotiff
from .model import ResidualScale, build_sen2_ssir, load_sen2_ssir_weights
from .patches import prepare_training_scenes, sample_training_patches
from .sentinel2 import discover_scenes, read_sentinel2_l2a, require_single_scene
from .training import train_sen2_ssir

__all__ = [
    "EVALUATION_RESULTS_DIR",
    "INFERENCE_RESULTS_DIR",
    "PREPARED_DATA_DIR",
    "PRETRAINED_WEIGHTS",
    "TESTING_SCENE_DIR",
    "TRAINING_RESULTS_DIR",
    "TRAINING_SCENE_DIR",
    "InferenceConfig",
    "PatchConfig",
    "ResidualScale",
    "TrainingConfig",
    "build_sen2_ssir",
    "discover_scenes",
    "ensure_project_directories",
    "evaluate_scene",
    "load_sen2_ssir_weights",
    "predict_tiled_to_npy",
    "prepare_training_scenes",
    "read_sentinel2_l2a",
    "require_single_scene",
    "sample_training_patches",
    "superresolve_scene_to_geotiff",
    "train_sen2_ssir",
]

__version__ = "1.0.0"
