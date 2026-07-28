"""Loss functions for imbalanced multi-class classification."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from dr.config import TrainingConfig


@tf.keras.utils.register_keras_serializable(package="dr")
class FocalLoss(tf.keras.losses.Loss):
    """Multi-class focal loss with optional per-class weights.

    Down-weights easy examples by ``(1 - p)^gamma`` so training focuses on the
    rare, hard-to-separate severity grades.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        class_weights: Optional[Sequence[float]] = None,
        name: str = "focal_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = (
            None if class_weights is None else [float(w) for w in class_weights]
        )

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = self.alpha * tf.pow(1.0 - y_pred, self.gamma) * cross_entropy

        if self.class_weights is not None:
            weights = tf.constant(self.class_weights, dtype=y_pred.dtype)
            per_sample_weight = tf.reduce_sum(
                y_true * weights, axis=-1, keepdims=True
            )
            loss = loss * per_sample_weight

        return tf.reduce_sum(loss, axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "class_weights": self.class_weights,
            }
        )
        return config


def build_loss(
    config: TrainingConfig, class_weights: Optional[np.ndarray] = None
):
    """Return the configured loss.

    Focal loss absorbs the class weights directly; plain cross-entropy relies
    on Keras' ``class_weight`` argument in ``fit`` instead.
    """
    if not config.use_focal_loss:
        print("Loss: categorical cross-entropy")
        return "categorical_crossentropy"

    weights = class_weights if config.use_class_weights else None
    print(
        "Loss: focal "
        f"(alpha={config.focal_alpha}, gamma={config.focal_gamma}, "
        f"class_weights={'on' if weights is not None else 'off'})"
    )
    return FocalLoss(
        alpha=config.focal_alpha, gamma=config.focal_gamma, class_weights=weights
    )
