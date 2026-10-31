"""Shared constants and configuration objects for Sen2-SSIR."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


MODEL_NAME = "Sen2-SSIR"
MODEL_SCALE = 2000.0
OUTPUT_SCALE = 10000.0
RESIDUAL_SCALE = 0.2

# GDAL's Sentinel-2 L2A 10 m subdataset exposes these bands in this order.
# Keeping this exact order is required by the published pretrained weights.
FINE_BANDS_MODEL_ORDER: Tuple[str, ...] = ("B4", "B3", "B2", "B8")
COARSE_BANDS_MODEL_ORDER: Tuple[str, ...] = (
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
)

# Order used by the final ten-band GeoTIFF.
OUTPUT_BANDS: Tuple[str, ...] = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)

BAND_WAVELENGTH_NM = {
    "B2": 490,
    "B3": 560,
    "B4": 665,
    "B5": 705,
    "B6": 740,
    "B7": 783,
    "B8": 842,
    "B8A": 865,
    "B11": 1610,
    "B12": 2190,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
TRAINING_SCENE_DIR = DATA_ROOT / "Training_Scene"
TESTING_SCENE_DIR = DATA_ROOT / "Testing_Scene"
PREPARED_DATA_DIR = DATA_ROOT / "Prepared_Data"
RESULTS_ROOT = PROJECT_ROOT / "Results"
TRAINING_RESULTS_DIR = RESULTS_ROOT / "Training"
EVALUATION_RESULTS_DIR = RESULTS_ROOT / "Evaluation"
INFERENCE_RESULTS_DIR = RESULTS_ROOT / "Inference"
PRETRAINED_WEIGHTS = PROJECT_ROOT / "weights" / "Sen2-SSIR_pretrained.weights.h5"


@dataclass(frozen=True)
class PatchConfig:
    """Patch geometry."""

    training_patch_size: int = 32
    patches_per_scene: int = 8000
    inference_patch_size: int = 128
    inference_border: int = 4
    seed: int = 42

    def validate(self) -> None:
        if self.training_patch_size <= 0 or self.training_patch_size % 2:
            raise ValueError("training_patch_size must be a positive even integer")
        if self.patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive")
        if self.inference_patch_size <= 0 or self.inference_patch_size % 2:
            raise ValueError("inference_patch_size must be a positive even integer")
        if self.inference_border < 0 or self.inference_border % 2:
            raise ValueError("inference_border must be a non-negative even integer")
        if self.inference_border * 2 >= self.inference_patch_size:
            raise ValueError("inference_border is too large for inference_patch_size")


@dataclass(frozen=True)
class TrainingConfig:
    """Default training settings."""

    learning_rate: float = 1e-4
    batch_size: int = 128
    epochs: int = 1500
    validation_fraction: float = 0.10
    early_stopping_patience: int = 50
    reduce_lr_patience: int = 15
    reduce_lr_cooldown: int = 20
    minimum_learning_rate: float = 1e-5
    seed: int = 42

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")


@dataclass(frozen=True)
class InferenceConfig:
    """Tiled-inference settings."""

    patch_size: int = 128
    border: int = 4
    batch_size: int = 32
    output_scale: float = OUTPUT_SCALE
    geotiff_dtype: str = "float32"
    geotiff_compression: str = "DEFLATE"
    geotiff_compression_level: int = 9

    def validate(self) -> None:
        if self.patch_size <= 0 or self.patch_size % 2:
            raise ValueError("patch_size must be a positive even integer")
        if self.border < 0 or self.border % 2:
            raise ValueError("border must be a non-negative even integer")
        if self.border * 2 >= self.patch_size:
            raise ValueError("border is too large for patch_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.output_scale <= 0:
            raise ValueError("output_scale must be positive")
        if self.geotiff_dtype.lower() not in {"float32", "uint16"}:
            raise ValueError("geotiff_dtype must be 'float32' or 'uint16'")
        if self.geotiff_compression.upper() not in {
            "NONE",
            "DEFLATE",
            "ZSTD",
            "LZW",
        }:
            raise ValueError(
                "geotiff_compression must be NONE, DEFLATE, ZSTD, or LZW"
            )
        if not 1 <= self.geotiff_compression_level <= 9:
            raise ValueError("geotiff_compression_level must be between 1 and 9")


def ensure_project_directories() -> None:
    """Create generated-data and results directories when they are absent."""

    for directory in (
        TRAINING_SCENE_DIR,
        TESTING_SCENE_DIR,
        PREPARED_DATA_DIR,
        TRAINING_RESULTS_DIR,
        EVALUATION_RESULTS_DIR,
        INFERENCE_RESULTS_DIR,
        PRETRAINED_WEIGHTS.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
