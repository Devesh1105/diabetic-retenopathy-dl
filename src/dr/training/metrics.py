"""Custom Keras metrics."""

from __future__ import annotations

from typing import List

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="dr")
class BalancedAccuracy(tf.keras.metrics.Metric):
    """Mean per-class recall, accumulated over a running confusion matrix.

    Plain accuracy is dominated by the No_DR class here, so this is the metric
    to watch when comparing runs.
    """

    def __init__(
        self, num_classes: int, name: str = "balanced_accuracy", **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        batch_cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        self.confusion_matrix.assign_add(batch_cm)

    def result(self):
        support = tf.reduce_sum(self.confusion_matrix, axis=1)
        correct = tf.linalg.diag_part(self.confusion_matrix)
        # Classes absent from the epoch contribute 0 rather than NaN.
        per_class_recall = tf.math.divide_no_nan(correct, support)
        seen = tf.cast(tf.math.count_nonzero(support), tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(per_class_recall), seen)

    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self) -> dict:
        config = super().get_config()
        config["num_classes"] = self.num_classes
        return config


def build_metrics(num_classes: int) -> List:
    """Metrics tracked for every run."""
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        BalancedAccuracy(num_classes=num_classes),
    ]


CUSTOM_OBJECTS = {"BalancedAccuracy": BalancedAccuracy}
"""Passed to ``load_model`` so saved checkpoints deserialise.

``dr.training.trainer.load_checkpoint`` extends this with ``FocalLoss`` and the
custom layers.
"""
