"""Smoke tests for the model, loss and metric.

Skipped automatically when TensorFlow is not installed. The encoder is built
without pretrained weights so the tests need no network access.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from dr.config import DataConfig, ModelConfig, TrainingConfig  # noqa: E402
from dr.models import ChannelPool, build_attention_unet  # noqa: E402
from dr.training.losses import FocalLoss, build_loss  # noqa: E402
from dr.training.metrics import BalancedAccuracy, build_metrics  # noqa: E402

IMG_SIZE = 64


@pytest.fixture(scope="module")
def model():
    data = DataConfig(img_size=IMG_SIZE, num_classes=5)
    return build_attention_unet(
        ModelConfig(pretrained_encoder=False, dense_units=32), data
    )


def test_model_output_shape_matches_class_count(model) -> None:
    batch = np.zeros((2, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    predictions = model.predict(batch, verbose=0)

    assert predictions.shape == (2, 5)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


def test_attention_and_decoder_layers_are_named_for_visualisation(model) -> None:
    names = {layer.name for layer in model.layers}

    for level in (2, 3, 4, 5):
        assert f"att{level}_multiply" in names
        assert f"d{level}_relu2" in names
    assert {"cbam_spatial_multiply", "global_avg_pool", "predictions"} <= names


def test_unsupported_encoder_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported encoder"):
        build_attention_unet(
            ModelConfig(encoder_name="resnet999", pretrained_encoder=False),
            DataConfig(img_size=IMG_SIZE),
        )


def test_focal_loss_penalises_confident_mistakes_more() -> None:
    loss = FocalLoss(alpha=0.25, gamma=2.0)
    y_true = tf.constant([[0.0, 1.0]])

    confident_correct = loss(y_true, tf.constant([[0.02, 0.98]]))
    confident_wrong = loss(y_true, tf.constant([[0.98, 0.02]]))

    assert float(confident_wrong) > float(confident_correct)


def test_focal_loss_class_weights_scale_the_loss() -> None:
    y_true = tf.constant([[0.0, 1.0]])
    y_pred = tf.constant([[0.4, 0.6]])

    unweighted = float(FocalLoss(class_weights=[1.0, 1.0])(y_true, y_pred))
    weighted = float(FocalLoss(class_weights=[1.0, 3.0])(y_true, y_pred))

    assert weighted == pytest.approx(unweighted * 3.0, rel=1e-5)


def test_focal_loss_survives_a_config_roundtrip() -> None:
    original = FocalLoss(alpha=0.3, gamma=1.5, class_weights=[1.0, 2.0])

    restored = FocalLoss.from_config(original.get_config())

    assert (restored.alpha, restored.gamma) == (0.3, 1.5)
    assert restored.class_weights == [1.0, 2.0]


def test_balanced_accuracy_ignores_class_frequency() -> None:
    # 9 majority samples all correct, 1 minority sample wrong.
    y_true = tf.constant([[1.0, 0.0]] * 9 + [[0.0, 1.0]])
    y_pred = tf.constant([[1.0, 0.0]] * 10)

    metric = BalancedAccuracy(num_classes=2)
    metric.update_state(y_true, y_pred)

    # Plain accuracy would be 0.9; mean per-class recall is (1.0 + 0.0) / 2.
    assert float(metric.result()) == pytest.approx(0.5)


def test_balanced_accuracy_skips_classes_absent_from_the_split() -> None:
    y_true = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y_pred = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    metric = BalancedAccuracy(num_classes=3)
    metric.update_state(y_true, y_pred)

    # Only class 0 appears, and it is perfectly predicted.
    assert float(metric.result()) == pytest.approx(1.0)


def test_balanced_accuracy_resets_between_epochs() -> None:
    metric = BalancedAccuracy(num_classes=2)
    metric.update_state(tf.constant([[1.0, 0.0]]), tf.constant([[0.0, 1.0]]))
    metric.reset_state()
    metric.update_state(tf.constant([[1.0, 0.0]]), tf.constant([[1.0, 0.0]]))

    assert float(metric.result()) == pytest.approx(1.0)


def test_saved_model_reloads_without_unsafe_deserialisation(model, tmp_path) -> None:
    """Regression test: CBAM once used Lambda layers, which Keras refuses to
    deserialise, leaving every checkpoint unloadable."""
    checkpoint = tmp_path / "model.keras"
    model.save(checkpoint)

    restored = tf.keras.models.load_model(
        checkpoint, custom_objects={"ChannelPool": ChannelPool}, compile=False
    )

    batch = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    assert np.allclose(
        restored.predict(batch, verbose=0), model.predict(batch, verbose=0), atol=1e-5
    )


def test_channel_pool_reduces_over_the_channel_axis() -> None:
    x = tf.constant(np.arange(24, dtype="float32").reshape(1, 2, 4, 3))

    mean_out = ChannelPool(reduction="mean")(x)
    max_out = ChannelPool(reduction="max")(x)

    assert mean_out.shape == (1, 2, 4, 1)
    assert np.allclose(mean_out.numpy(), x.numpy().mean(axis=-1, keepdims=True))
    assert np.allclose(max_out.numpy(), x.numpy().max(axis=-1, keepdims=True))


def test_channel_pool_rejects_an_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="reduction must be"):
        ChannelPool(reduction="median")


def test_model_trains_for_one_step(model) -> None:
    """Compile-and-fit smoke test: the graph, loss and metrics fit together."""
    config = TrainingConfig(use_focal_loss=True, use_class_weights=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=build_loss(config, np.ones(5)),
        metrics=build_metrics(5),
    )

    x = np.random.default_rng(0).random((4, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    y = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=5)

    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)

    assert "balanced_accuracy" in history.history
    assert np.isfinite(history.history["loss"][0])
