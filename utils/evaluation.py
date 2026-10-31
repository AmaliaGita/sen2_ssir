"""End-to-end simulated evaluation for one Sentinel-2 L2A scene."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .config import InferenceConfig
from .inference import predict_tiled_to_npy
from .metrics import evaluate_six_bands
from .preprocessing import prepare_simulated_inputs
from .sentinel2 import PixelRoi, read_sentinel2_l2a, validate_scene_shapes


def evaluate_scene(
    model,
    scene_path,
    results_directory,
    config: InferenceConfig = InferenceConfig(),
    roi: Optional[PixelRoi] = None,
    verbose: bool = True,
):
    """Run the manuscript's simulated 20 m/40 m assessment protocol."""

    config.validate()
    scene = read_sentinel2_l2a(scene_path, roi=roi)
    validate_scene_shapes(scene)
    metadata = scene.metadata
    simulated = prepare_simulated_inputs(scene.fine, scene.coarse)
    del scene

    results_directory = Path(results_directory).expanduser().resolve()
    results_directory.mkdir(parents=True, exist_ok=True)
    roi_suffix = ""
    if roi is not None:
        roi_suffix = "_roi_" + "_".join(str(int(value)) for value in roi)
    result_stem = f"{metadata.scene_id}{roi_suffix}"
    prediction_path = results_directory / f"{result_stem}_simulated_prediction.npy"
    metrics_path = results_directory / f"{result_stem}_metrics.json"

    predict_tiled_to_npy(
        model,
        simulated.fine,
        simulated.coarse,
        prediction_path,
        config=config,
        verbose=verbose,
    )
    prediction = np.load(prediction_path, mmap_mode="r")
    metrics = evaluate_six_bands(simulated.target, prediction)
    metrics["model"] = "Sen2-SSIR"
    metrics["scene_id"] = metadata.scene_id
    metrics["source"] = metadata.source
    metrics["roi_10m_pixels"] = list(roi) if roi is not None else None
    metrics["protocol"] = "native 20 m target; simulated 20 m fine input and 40 m coarse input"
    metrics["prediction_path"] = str(prediction_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, prediction_path, metrics_path
