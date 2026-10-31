"""Preprocessing for Sen2-SSIR."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from skimage.transform import resize


@dataclass
class SimulatedInputs:
    """Arrays used by the 2x simulated-resolution protocol."""

    fine: np.ndarray
    coarse: np.ndarray
    target: np.ndarray


def legacy_downsample(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """Apply the Gaussian-plus-pixel-aggregation degradation.

    Preserves the numerical preprocessing used to train the
    published Sen2-SSIR checkpoint: Gaussian sigma ``1 / scale`` followed by
    non-overlapping mean aggregation.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    image = np.asarray(image)
    squeeze = image.ndim == 2
    if squeeze:
        image = image[..., np.newaxis]
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width[, channels])")

    output_height = image.shape[0] // scale
    output_width = image.shape[1] // scale
    if output_height == 0 or output_width == 0:
        raise ValueError("image is smaller than the requested scale")

    usable_height = output_height * scale
    usable_width = output_width * scale
    image = image[:usable_height, :usable_width]

    result = np.empty(
        (output_height, output_width, image.shape[-1]),
        dtype=np.float32,
    )
    for band_index in range(image.shape[-1]):
        blurred = gaussian_filter(
            image[..., band_index],
            sigma=1.0 / scale,
        )
        result[..., band_index] = block_reduce(
            blurred,
            block_size=(scale, scale),
            func=np.mean,
        ).astype(np.float32, copy=False)
    return result[..., 0] if squeeze else result


def legacy_upsample(
    image: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """Bilinearly resize each band"""

    image = np.asarray(image)
    squeeze = image.ndim == 2
    if squeeze:
        image = image[..., np.newaxis]
    if image.ndim != 3:
        raise ValueError("image must have shape (height, width[, channels])")
    if len(output_shape) != 2 or min(output_shape) <= 0:
        raise ValueError("output_shape must contain two positive dimensions")

    result = np.empty((*output_shape, image.shape[-1]), dtype=np.float32)
    for band_index in range(image.shape[-1]):
        # Division by 30000 is retained to reproduce the previous skimage path.
        result[..., band_index] = (
            resize(
                image[..., band_index] / 30000.0,
                output_shape,
                mode="reflect",
                order=1,
                anti_aliasing=False,
            )
            * 30000.0
        ).astype(np.float32, copy=False)
    return result[..., 0] if squeeze else result


def prepare_simulated_inputs(
    native_fine: np.ndarray,
    native_coarse: np.ndarray,
) -> SimulatedInputs:
    """Create the manuscript's simulated 20 m/40 m training configuration."""

    if native_fine.ndim != 3 or native_fine.shape[-1] != 4:
        raise ValueError("native_fine must have shape (height, width, 4)")
    if native_coarse.ndim != 3 or native_coarse.shape[-1] != 6:
        raise ValueError("native_coarse must have shape (height, width, 6)")

    fine = legacy_downsample(native_fine, scale=2)
    # Preserve the original pipeline's operation order: blur and aggregate the
    # native integer data before converting the 20 m reference to float32.
    coarse_low = legacy_downsample(native_coarse, scale=2)
    target = np.asarray(native_coarse, dtype=np.float32)

    common_height = min(fine.shape[0], target.shape[0])
    common_width = min(fine.shape[1], target.shape[1])
    common_height -= common_height % 2
    common_width -= common_width % 2
    if common_height <= 0 or common_width <= 0:
        raise ValueError("scene is too small for simulated evaluation")

    fine = fine[:common_height, :common_width]
    target = target[:common_height, :common_width]
    coarse_low = coarse_low[: common_height // 2, : common_width // 2]
    return SimulatedInputs(fine=fine, coarse=coarse_low, target=target)
