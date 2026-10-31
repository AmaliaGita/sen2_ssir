"""Dataset Preparation"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .config import MODEL_SCALE, TrainingConfig


@dataclass(frozen=True)
class PreparedScene:
    """Paths and size of one prepared patch dataset."""

    directory: Path
    fine_path: Path
    coarse_path: Path
    target_path: Path
    patch_count: int


def discover_prepared_scenes(prepared_directory) -> List[PreparedScene]:
    """Validate and list per-scene datasets produced by ``patches.py``."""

    prepared_directory = Path(prepared_directory).expanduser().resolve()
    if not prepared_directory.is_dir():
        raise FileNotFoundError(
            f"Prepared-data directory does not exist: {prepared_directory}"
        )

    scenes: List[PreparedScene] = []
    for directory in sorted(path for path in prepared_directory.iterdir() if path.is_dir()):
        paths = [directory / name for name in ("fine.npy", "coarse.npy", "target.npy")]
        if not all(path.is_file() for path in paths):
            continue

        shapes = [np.load(path, mmap_mode="r").shape for path in paths]
        if any(len(shape) != 4 for shape in shapes):
            raise ValueError(f"Prepared arrays must be four-dimensional: {directory}")
        if not (shapes[0][0] == shapes[1][0] == shapes[2][0]):
            raise ValueError(f"Patch counts do not match in {directory}")
        if shapes[0][-1] != 4 or shapes[1][-1] != 6 or shapes[2][-1] != 6:
            raise ValueError(f"Prepared band counts are invalid in {directory}")
        if shapes[0][1:3] != shapes[1][1:3] or shapes[0][1:3] != shapes[2][1:3]:
            raise ValueError(f"Prepared patch dimensions do not match in {directory}")

        scenes.append(
            PreparedScene(
                directory=directory,
                fine_path=paths[0],
                coarse_path=paths[1],
                target_path=paths[2],
                patch_count=shapes[0][0],
            )
        )

    if not scenes:
        raise RuntimeError(
            f"No prepared scenes found in {prepared_directory}. "
            "Run prepare_training_scenes first."
        )
    return scenes


def split_patch_indices(
    scenes: Sequence[PreparedScene],
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation references ``(scene, patch)``."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    references = np.concatenate(
        [
            np.column_stack(
                (
                    np.full(scene.patch_count, scene_index, dtype=np.int32),
                    np.arange(scene.patch_count, dtype=np.int32),
                )
            )
            for scene_index, scene in enumerate(scenes)
        ],
        axis=0,
    )
    if len(references) < 2:
        raise ValueError("At least two prepared patches are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(references)
    validation_count = max(1, int(round(len(references) * validation_fraction)))
    validation_count = min(validation_count, len(references) - 1)
    return references[validation_count:], references[:validation_count]


class PatchSequence(tf.keras.utils.Sequence):
    """Keras sequence that reads only the current batch from NPY memmaps."""

    def __init__(
        self,
        scenes: Sequence[PreparedScene],
        references: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.scenes = list(scenes)
        self.references = np.asarray(references, dtype=np.int32).copy()
        if self.references.ndim != 2 or self.references.shape[1] != 2:
            raise ValueError("references must have shape (number_of_patches, 2)")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self._arrays = [
            (
                np.load(scene.fine_path, mmap_mode="r"),
                np.load(scene.coarse_path, mmap_mode="r"),
                np.load(scene.target_path, mmap_mode="r"),
            )
            for scene in self.scenes
        ]
        if self.shuffle:
            self.rng.shuffle(self.references)

    def __len__(self) -> int:
        return int(np.ceil(len(self.references) / float(self.batch_size)))

    def __getitem__(self, batch_index: int):
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.references))
        if start < 0 or start >= len(self.references):
            raise IndexError(batch_index)
        batch_references = self.references[start:stop]

        fine = np.stack(
            [self._arrays[scene_index][0][patch_index] for scene_index, patch_index in batch_references]
        ).astype(np.float32, copy=False)
        coarse = np.stack(
            [self._arrays[scene_index][1][patch_index] for scene_index, patch_index in batch_references]
        ).astype(np.float32, copy=False)
        target = np.stack(
            [self._arrays[scene_index][2][patch_index] for scene_index, patch_index in batch_references]
        ).astype(np.float32, copy=False)
        return [fine / MODEL_SCALE, coarse / MODEL_SCALE], target / MODEL_SCALE

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.references)


def create_training_sequences(
    prepared_directory,
    config: TrainingConfig = TrainingConfig(),
) -> Tuple[PatchSequence, PatchSequence]:
    """Create deterministic memory-mapped train and validation sequences."""

    config.validate()
    scenes = discover_prepared_scenes(prepared_directory)
    train_indices, validation_indices = split_patch_indices(
        scenes,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_sequence = PatchSequence(
        scenes,
        train_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_sequence = PatchSequence(
        scenes,
        validation_indices,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    return train_sequence, validation_sequence
