"""Training patch preparation and memory-safe inference tiling."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from .config import MODEL_SCALE, PatchConfig
from .preprocessing import SimulatedInputs, legacy_upsample, prepare_simulated_inputs
from .sentinel2 import discover_scenes, read_sentinel2_l2a, validate_scene_shapes


@dataclass(frozen=True)
class TilePosition:
    y: int
    x: int
    height: int
    width: int


@dataclass
class TileBatch:
    fine: np.ndarray
    coarse: np.ndarray
    positions: List[TilePosition]


def sample_training_patches(
    simulated: SimulatedInputs,
    number_of_patches: int = 8000,
    patch_size: int = 32,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample aligned 20 m/40 m patches"""

    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size must be a positive even integer")
    if number_of_patches <= 0:
        raise ValueError("number_of_patches must be positive")

    coarse_patch_size = patch_size // 2
    if min(simulated.coarse.shape[:2]) < coarse_patch_size:
        raise ValueError("The scene is smaller than one requested training patch")

    fine_patches = np.empty(
        (number_of_patches, patch_size, patch_size, 4), dtype=np.float32
    )
    coarse_patches = np.empty(
        (number_of_patches, patch_size, patch_size, 6), dtype=np.float32
    )
    targets = np.empty(
        (number_of_patches, patch_size, patch_size, 6), dtype=np.float32
    )

    maximum_y = simulated.coarse.shape[0] - coarse_patch_size
    maximum_x = simulated.coarse.shape[1] - coarse_patch_size
    rng = np.random.default_rng(seed)

    for patch_index in range(number_of_patches):
        low_y = int(rng.integers(0, maximum_y + 1))
        low_x = int(rng.integers(0, maximum_x + 1))
        high_y = low_y * 2
        high_x = low_x * 2

        fine_patches[patch_index] = simulated.fine[
            high_y : high_y + patch_size,
            high_x : high_x + patch_size,
        ]
        targets[patch_index] = simulated.target[
            high_y : high_y + patch_size,
            high_x : high_x + patch_size,
        ]
        coarse_patch = simulated.coarse[
            low_y : low_y + coarse_patch_size,
            low_x : low_x + coarse_patch_size,
        ]
        coarse_patches[patch_index] = legacy_upsample(
            coarse_patch,
            (patch_size, patch_size),
        )

    return fine_patches, coarse_patches, targets


def _write_scene_metadata(path: Path, metadata: Dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_training_scenes(
    scene_directory,
    prepared_directory,
    patch_config: PatchConfig = PatchConfig(),
    overwrite: bool = False,
) -> List[Path]:
    """Prepare per-scene NPY patch datasets for Notebook 1."""

    patch_config.validate()
    scenes = discover_scenes(scene_directory)
    if not scenes:
        raise RuntimeError(f"No Sentinel-2 L2A scenes found in {scene_directory}")

    prepared_directory = Path(prepared_directory).expanduser().resolve()
    prepared_directory.mkdir(parents=True, exist_ok=True)
    prepared_scenes: List[Path] = []

    for scene_index, scene_path in enumerate(scenes):
        scene = read_sentinel2_l2a(scene_path)
        validate_scene_shapes(scene)
        metadata = scene.metadata
        output_directory = prepared_directory / metadata.scene_id
        required_files = (
            output_directory / "fine.npy",
            output_directory / "coarse.npy",
            output_directory / "target.npy",
            output_directory / "metadata.json",
        )
        if not overwrite and all(path.is_file() for path in required_files):
            prepared_scenes.append(output_directory)
            continue

        output_directory.mkdir(parents=True, exist_ok=True)
        simulated = prepare_simulated_inputs(scene.fine, scene.coarse)
        del scene
        fine, coarse, target = sample_training_patches(
            simulated,
            number_of_patches=patch_config.patches_per_scene,
            patch_size=patch_config.training_patch_size,
            seed=patch_config.seed + scene_index,
        )
        np.save(output_directory / "fine.npy", fine)
        np.save(output_directory / "coarse.npy", coarse)
        np.save(output_directory / "target.npy", target)
        _write_scene_metadata(
            output_directory / "metadata.json",
            {
                "model": "Sen2-SSIR",
                "source_scene": metadata.source,
                "scene_id": metadata.scene_id,
                "fine_shape": [metadata.height, metadata.width, 4],
                "coarse_shape": [metadata.height // 2, metadata.width // 2, 6],
                "patches": patch_config.patches_per_scene,
                "patch_size": patch_config.training_patch_size,
                "seed": patch_config.seed + scene_index,
                "normalization_scale": MODEL_SCALE,
                "preprocessing": "legacy Gaussian degradation, mean aggregation, bilinear upsampling",
            },
        )
        prepared_scenes.append(output_directory)

    manifest = {
        "model": "Sen2-SSIR",
        "patch_config": asdict(patch_config),
        "scenes": [str(path) for path in prepared_scenes],
    }
    _write_scene_metadata(prepared_directory / "manifest.json", manifest)
    return prepared_scenes


def _output_starts(length: int, interior_size: int) -> List[int]:
    if length <= 0:
        raise ValueError("Image dimensions must be positive")
    if length <= interior_size:
        return [0]
    count = int(np.ceil(length / float(interior_size)))
    return list(
        dict.fromkeys(
            min(index * interior_size, length - interior_size)
            for index in range(count)
        )
    )


def count_inference_tiles(
    image_shape: Tuple[int, int],
    patch_size: int,
    border: int,
) -> int:
    """Return the number of overlapping tiles required for an image."""

    interior = patch_size - 2 * border
    if interior <= 0:
        raise ValueError("border is too large for patch_size")
    return len(_output_starts(image_shape[0], interior)) * len(
        _output_starts(image_shape[1], interior)
    )


def iter_inference_batches(
    fine: np.ndarray,
    coarse: np.ndarray,
    patch_size: int = 128,
    border: int = 4,
    batch_size: int = 1,
) -> Iterator[TileBatch]:
    """Yield aligned, normalized NHWC tiles without materializing all patches."""

    if fine.ndim != 3 or fine.shape[-1] != 4:
        raise ValueError("fine must have shape (height, width, 4)")
    if coarse.ndim != 3 or coarse.shape[-1] != 6:
        raise ValueError("coarse must have shape (height/2, width/2, 6)")
    if fine.shape[0] != coarse.shape[0] * 2 or fine.shape[1] != coarse.shape[1] * 2:
        raise ValueError("fine and coarse inputs must have a 2x spatial relationship")
    if patch_size % 2 or border % 2:
        raise ValueError("patch_size and border must be even")
    if border * 2 >= patch_size or batch_size <= 0:
        raise ValueError("Invalid patch, border, or batch size")

    interior = patch_size - 2 * border
    coarse_patch_size = patch_size // 2
    coarse_border = border // 2
    y_starts = _output_starts(fine.shape[0], interior)
    x_starts = _output_starts(fine.shape[1], interior)

    extra_y = max(0, patch_size - (fine.shape[0] + 2 * border))
    extra_x = max(0, patch_size - (fine.shape[1] + 2 * border))
    fine_padded = np.pad(
        fine,
        (
            (border, border + extra_y),
            (border, border + extra_x),
            (0, 0),
        ),
        mode="symmetric",
    )
    coarse_padded = np.pad(
        coarse,
        (
            (coarse_border, coarse_border + extra_y // 2),
            (coarse_border, coarse_border + extra_x // 2),
            (0, 0),
        ),
        mode="symmetric",
    )

    batch_fine: List[np.ndarray] = []
    batch_coarse: List[np.ndarray] = []
    batch_positions: List[TilePosition] = []

    for y in y_starts:
        for x in x_starts:
            fine_patch = fine_padded[y : y + patch_size, x : x + patch_size]
            coarse_patch = coarse_padded[
                y // 2 : y // 2 + coarse_patch_size,
                x // 2 : x // 2 + coarse_patch_size,
            ]
            coarse_upsampled = legacy_upsample(
                coarse_patch,
                (patch_size, patch_size),
            )
            batch_fine.append(fine_patch.astype(np.float32, copy=False) / MODEL_SCALE)
            batch_coarse.append(coarse_upsampled / MODEL_SCALE)
            batch_positions.append(
                TilePosition(
                    y=y,
                    x=x,
                    height=min(interior, fine.shape[0] - y),
                    width=min(interior, fine.shape[1] - x),
                )
            )

            if len(batch_fine) == batch_size:
                yield TileBatch(
                    fine=np.stack(batch_fine),
                    coarse=np.stack(batch_coarse),
                    positions=batch_positions,
                )
                batch_fine, batch_coarse, batch_positions = [], [], []

    if batch_fine:
        yield TileBatch(
            fine=np.stack(batch_fine),
            coarse=np.stack(batch_coarse),
            positions=batch_positions,
        )
