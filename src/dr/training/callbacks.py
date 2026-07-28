"""Keras callbacks shared by every run."""

from __future__ import annotations

from typing import List

from tensorflow.keras import callbacks as keras_callbacks

from dr.config import ExperimentConfig


def build_callbacks(config: ExperimentConfig) -> List[keras_callbacks.Callback]:
    """Checkpointing, early stopping, LR schedule and logging.

    All artefacts land under ``paths.models`` (checkpoint) and
    ``paths.results`` (TensorBoard logs, CSV history).
    """
    config.paths.create()
    training = config.training

    return [
        keras_callbacks.ModelCheckpoint(
            filepath=str(config.paths.models / training.checkpoint_filename),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=training.reduce_lr_factor,
            patience=training.reduce_lr_patience,
            min_lr=training.min_learning_rate,
            verbose=1,
        ),
        keras_callbacks.TensorBoard(
            log_dir=str(config.paths.results / "logs"),
            histogram_freq=1,
            write_graph=True,
        ),
        keras_callbacks.CSVLogger(
            str(config.paths.results / "training_log.csv"), append=False
        ),
    ]
