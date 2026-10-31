"""Metrics for simulated Sen2-SSIR evaluation."""

import math
from typing import Dict

import numpy as np

from .config import COARSE_BANDS_MODEL_ORDER


MANUSCRIPT_ERGAS_RATIO = 0.25


def _validate_pair(reference: np.ndarray, prediction: np.ndarray) -> None:
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Reference and prediction shapes differ: {reference.shape} vs {prediction.shape}"
        )
    if reference.ndim != 3 or reference.shape[-1] != 6:
        raise ValueError("Evaluation arrays must have shape (height, width, 6)")


def _sre_db_from_band_statistics(
    band_means: np.ndarray,
    band_mse: np.ndarray,
) -> np.ndarray:
    """Return SRE in dB for every spectral band.

    For a ground-truth band ``x`` and prediction ``y`` this implements
    ``10 log10(mean(x)^2 / mean((x - y)^2))``. The multi-band result is the
    arithmetic mean of these per-band dB values.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.square(band_means) / band_mse)


def evaluate_six_bands(
    reference: np.ndarray,
    prediction: np.ndarray,
    chunk_rows: int = 256,
) -> Dict:
    """Calculate the three aggregate metrics.

    RMSE is calculated globally over the complete six-band cube. ERGAS follows
    ``sewar.full_ref.ergas`` with its default resolution ratio ``r=0.25``.
    SRE uses ``10 log10(mean(reference)^2 / MSE)`` separately for each band 
    and is aggregated by averaging the six band-level
    dB values. Float64 chunk accumulation keeps all metrics memory-efficient.
    """

    _validate_pair(reference, prediction)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    total_count = 0
    total_squared_error = 0.0
    band_counts = np.zeros(reference.shape[-1], dtype=np.int64)
    band_reference_sums = np.zeros(reference.shape[-1], dtype=np.float64)
    band_squared_error_sums = np.zeros(reference.shape[-1], dtype=np.float64)

    for y0 in range(0, reference.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, reference.shape[0])
        reference_chunk = np.asarray(reference[y0:y1], dtype=np.float64)
        prediction_chunk = np.asarray(prediction[y0:y1], dtype=np.float64)
        difference = reference_chunk - prediction_chunk
        squared_error = np.square(difference)

        total_count += reference_chunk.size
        total_squared_error += float(squared_error.sum(dtype=np.float64))
        spatial_axes = (0, 1)
        pixels_per_band = reference_chunk.shape[0] * reference_chunk.shape[1]
        band_counts += pixels_per_band
        band_reference_sums += reference_chunk.sum(axis=spatial_axes, dtype=np.float64)
        band_squared_error_sums += squared_error.sum(
            axis=spatial_axes,
            dtype=np.float64,
        )

    global_rmse = math.sqrt(total_squared_error / total_count)
    band_means = band_reference_sums / band_counts
    band_rmses = np.sqrt(band_squared_error_sums / band_counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_band_errors = band_rmses / band_means
    ergas = 100.0 * MANUSCRIPT_ERGAS_RATIO * math.sqrt(
        float(np.mean(np.square(relative_band_errors)))
    )
    band_sre_db = _sre_db_from_band_statistics(
        band_means,
        np.square(band_rmses),
    )
    per_band = {}
    for band_index, band_name in enumerate(COARSE_BANDS_MODEL_ORDER):
        per_band[band_name] = {
            "RMSE": float(band_rmses[band_index]),
            "SRE_dB": float(band_sre_db[band_index]),
            "ERGAS": float(
                100.0
                * MANUSCRIPT_ERGAS_RATIO
                * relative_band_errors[band_index]
            ),
        }

    return {
        "bands": list(COARSE_BANDS_MODEL_ORDER),
        "aggregate": {
            "RMSE": float(global_rmse),
            "SRE_dB": float(np.mean(band_sre_db)),
            "ERGAS": float(ergas),
        },
        "per_band": per_band,
        "settings": {
            "protocol": "Sen2-SSIR manuscript evaluation",
            "rmse_aggregation": "global over all pixels and six bands",
            "sre_definition": "10*log10(mean(reference)^2/MSE)",
            "sre_aggregation": "arithmetic mean of six per-band dB values",
            "ergas_resolution_ratio": MANUSCRIPT_ERGAS_RATIO,
        },
    }
