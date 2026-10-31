"""Plotting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history_csv):
    """Plot training and validation loss from the persistent CSV log."""

    data = np.genfromtxt(
        Path(history_csv),
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if data.size == 0:
        raise ValueError("The training-history file is empty")
    data = np.atleast_1d(data)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(data["epoch"], data["loss"], label="Training")
    if "val_loss" in data.dtype.names:
        axis.plot(data["epoch"], data["val_loss"], label="Validation")
    axis.set(title="Sen2-SSIR loss", xlabel="Epoch", ylabel="Mean absolute error")
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def plot_simulated_comparison(reference, prediction, band_index: int = 0):
    """Display one target/prediction/error band from simulated evaluation."""

    reference_band = np.asarray(reference[..., band_index])
    prediction_band = np.asarray(prediction[..., band_index])
    lower, upper = np.percentile(reference_band, (2, 98))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(reference_band, cmap="gray", vmin=lower, vmax=upper)
    axes[0].set_title("Reference")
    axes[1].imshow(prediction_band, cmap="gray", vmin=lower, vmax=upper)
    axes[1].set_title("Sen2-SSIR")
    error_plot = axes[2].imshow(prediction_band - reference_band, cmap="RdBu_r")
    axes[2].set_title("Prediction - reference")
    figure.colorbar(error_plot, ax=axes[2], fraction=0.046)
    for axis in axes:
        axis.axis("off")
    return figure
