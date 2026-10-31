"""Training Sen2-SSIR."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
from typing import Dict

import numpy as np
import tensorflow as tf

from .config import PRETRAINED_WEIGHTS, TrainingConfig
from .dataset import create_training_sequences
from .model import ResidualScale, build_sen2_ssir


TRAINING_MODES = ("scratch", "pretrained", "resume")


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable data ordering."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compile_sen2_ssir(model, learning_rate: float = 1e-4):
    """Compile the loss and optimizer"""

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        loss="mean_absolute_error",
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def _completed_epochs(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def initialize_training_model(
    mode: str,
    results_directory,
    config: TrainingConfig,
    pretrained_weights=PRETRAINED_WEIGHTS,
):
    """Initialize scratch, pretrained/fine-tuning, or resumable training."""

    mode = mode.lower().strip()
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES}")

    results_directory = Path(results_directory).expanduser().resolve()
    latest_model = results_directory / "Sen2-SSIR_latest.h5"
    if mode == "resume":
        if not latest_model.is_file():
            raise FileNotFoundError(
                f"Cannot resume because the latest model is absent: {latest_model}"
            )
        model = tf.keras.models.load_model(
            str(latest_model),
            custom_objects={"ResidualScale": ResidualScale},
        )
        if model.optimizer is None:
            compile_sen2_ssir(model, config.learning_rate)
        return model

    model = build_sen2_ssir()
    if mode == "pretrained":
        pretrained_weights = Path(pretrained_weights).expanduser().resolve()
        if not pretrained_weights.is_file():
            raise FileNotFoundError(
                f"Published pretrained weights not found: {pretrained_weights}"
            )
        model.load_weights(str(pretrained_weights))
    return compile_sen2_ssir(model, config.learning_rate)


def train_sen2_ssir(
    prepared_directory,
    results_directory,
    mode: str = "pretrained",
    config: TrainingConfig = TrainingConfig(),
    pretrained_weights=PRETRAINED_WEIGHTS,
    verbose: int = 1,
):
    """Train Sen2-SSIR and return the model and Keras History object.

    Modes are ``scratch`` (random initialization), ``pretrained`` (fine-tune
    the published checkpoint), and ``resume`` (continue the latest run while
    restoring its optimizer state).
    """

    config.validate()
    set_reproducible_seed(config.seed)
    results_directory = Path(results_directory).expanduser().resolve()
    results_directory.mkdir(parents=True, exist_ok=True)
    history_path = results_directory / "training_history.csv"
    initial_epoch = _completed_epochs(history_path) if mode.lower() == "resume" else 0
    if initial_epoch >= config.epochs:
        raise ValueError(
            f"The run already contains {initial_epoch} epochs; increase config.epochs "
            "to continue training."
        )

    train_sequence, validation_sequence = create_training_sequences(
        prepared_directory,
        config,
    )
    model = initialize_training_model(
        mode,
        results_directory,
        config,
        pretrained_weights=pretrained_weights,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(results_directory / "Sen2-SSIR_best.weights.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(results_directory / "Sen2-SSIR_latest.h5"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(history_path),
            append=mode.lower() == "resume",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.reduce_lr_patience,
            cooldown=config.reduce_lr_cooldown,
            min_lr=config.minimum_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    run_metadata: Dict = {
        "model": "Sen2-SSIR",
        "mode": mode.lower(),
        "initial_epoch": initial_epoch,
        "training_config": asdict(config),
        "prepared_directory": str(Path(prepared_directory).resolve()),
        "pretrained_weights": str(Path(pretrained_weights).resolve()),
    }
    (results_directory / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    history = model.fit(
        train_sequence,
        validation_data=validation_sequence,
        initial_epoch=initial_epoch,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    model.save_weights(str(results_directory / "Sen2-SSIR_final.weights.h5"))
    return model, history
