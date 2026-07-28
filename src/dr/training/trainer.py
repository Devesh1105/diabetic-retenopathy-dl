"""End-to-end training: build the model, fit it, save the artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from dr.config import ExperimentConfig
from dr.data.generators import Generators, compute_class_weights, create_generators
from dr.models.attention import ChannelPool
from dr.models.attention_unet import build_attention_unet
from dr.training.callbacks import build_callbacks
from dr.training.losses import FocalLoss, build_loss
from dr.training.metrics import CUSTOM_OBJECTS, build_metrics


def compile_model(
    model: tf.keras.Model,
    config: ExperimentConfig,
    class_weights: Optional[np.ndarray] = None,
) -> tf.keras.Model:
    """Attach the optimiser, loss and metrics described by ``config``."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.training.learning_rate
        ),
        loss=build_loss(config.training, class_weights),
        metrics=build_metrics(config.data.num_classes),
    )
    return model


def load_checkpoint(
    config: ExperimentConfig, class_weights: Optional[np.ndarray] = None
) -> tf.keras.Model:
    """Load the best checkpoint for a run, recompiling if deserialisation fails."""
    checkpoint = config.paths.models / config.training.checkpoint_filename
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    custom_objects = {
        **CUSTOM_OBJECTS,
        "FocalLoss": FocalLoss,
        "ChannelPool": ChannelPool,
    }
    try:
        return tf.keras.models.load_model(
            checkpoint, custom_objects=custom_objects, compile=True
        )
    except (ValueError, TypeError) as exc:
        print(f"Could not restore the compiled model ({exc}); recompiling.")
        model = tf.keras.models.load_model(
            checkpoint, custom_objects=custom_objects, compile=False
        )
        return compile_model(model, config, class_weights)


def train(
    config: ExperimentConfig,
    generators: Optional[Generators] = None,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Generators, Dict[int, float]]:
    """Run one experiment end to end.

    Returns the trained model (with the best weights restored), the Keras
    history, the generators and the class-weight mapping, so callers can go
    straight into evaluation.
    """
    config.paths.create()
    config.save(config.paths.results / "config.yaml")

    tf.keras.utils.set_random_seed(config.data.seed)

    if generators is None:
        generators = create_generators(config)

    class_weight_dict, class_weights = compute_class_weights(
        generators.train, config.data.class_names
    )

    print("\nBuilding Attention U-Net...")
    model = build_attention_unet(config.model, config.data)
    compile_model(model, config, class_weights)
    model.summary()
    print(f"Trainable parameters: {model.count_params():,}")

    # Focal loss already folds in the class weights, so applying Keras'
    # class_weight on top would double-count them.
    fit_class_weight = (
        class_weight_dict
        if config.training.use_class_weights and not config.training.use_focal_loss
        else None
    )

    print("\n" + "=" * 70)
    print(f"TRAINING: {config.name}")
    print("=" * 70)

    history = model.fit(
        generators.train,
        epochs=config.training.num_epochs,
        validation_data=generators.validation,
        class_weight=fit_class_weight,
        callbacks=build_callbacks(config),
        verbose=1,
    )

    print(f"\nTraining finished after {len(history.history['loss'])} epoch(s).")
    return model, history, generators, class_weight_dict
