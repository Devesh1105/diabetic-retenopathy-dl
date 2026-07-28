"""Attention building blocks: additive attention gates and CBAM."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="dr")
class ChannelPool(layers.Layer):
    """Reduce a feature map over its channel axis, keeping the axis.

    CBAM's spatial branch needs channel-wise mean and max maps. A ``Lambda``
    layer would do it, but Keras refuses to deserialise saved models
    containing a Python lambda, which makes the checkpoints unloadable. This
    layer serialises cleanly instead.
    """

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__(**kwargs)
        if reduction not in {"mean", "max"}:
            raise ValueError(f"reduction must be 'mean' or 'max', got {reduction!r}")
        self.reduction = reduction

    def call(self, inputs):
        op = tf.reduce_mean if self.reduction == "mean" else tf.reduce_max
        return op(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)

    def get_config(self) -> dict:
        config = super().get_config()
        config["reduction"] = self.reduction
        return config


def attention_gate(gating, skip_connection, inter_channels: int, name: str):
    """Additive attention gate (Oktay et al., 2018).

    The coarse decoder signal ``gating`` gates the finer encoder
    ``skip_connection``, suppressing background activations before the skip is
    concatenated into the decoder.

    Args:
        gating: Decoder feature map, already upsampled to the skip's resolution.
        skip_connection: Encoder feature map to be gated.
        inter_channels: Width of the shared intermediate projection.
        name: Prefix for the layer names.

    Returns:
        The skip connection scaled by the learned attention coefficients.
    """
    g = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name}_g_conv")(gating)
    g = layers.BatchNormalization(name=f"{name}_g_bn")(g)

    x = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name}_x_conv")(
        skip_connection
    )
    x = layers.BatchNormalization(name=f"{name}_x_bn")(x)

    psi = layers.Add(name=f"{name}_add")([g, x])
    psi = layers.Activation("relu", name=f"{name}_relu")(psi)
    psi = layers.Conv2D(1, 1, padding="same", name=f"{name}_psi_conv")(psi)
    psi = layers.BatchNormalization(name=f"{name}_psi_bn")(psi)
    psi = layers.Activation("sigmoid", name=f"{name}_sigmoid")(psi)

    return layers.Multiply(name=f"{name}_multiply")([skip_connection, psi])


def cbam_block(input_tensor, reduction: int = 16, name: str = "cbam"):
    """Convolutional Block Attention Module (Woo et al., 2018).

    Applies channel attention (shared MLP over average- and max-pooled
    descriptors) followed by spatial attention (7x7 conv over the pooled
    channel maps).
    """
    channels = input_tensor.shape[-1]

    # --- Channel attention ---
    avg_pool = layers.GlobalAveragePooling2D(name=f"{name}_avg_pool")(input_tensor)
    max_pool = layers.GlobalMaxPooling2D(name=f"{name}_max_pool")(input_tensor)

    shared_hidden = layers.Dense(
        max(channels // reduction, 1), activation="relu", name=f"{name}_dense1"
    )
    shared_output = layers.Dense(channels, name=f"{name}_dense2")

    channel_attention = layers.Add(name=f"{name}_channel_add")(
        [shared_output(shared_hidden(avg_pool)), shared_output(shared_hidden(max_pool))]
    )
    channel_attention = layers.Activation(
        "sigmoid", name=f"{name}_channel_sigmoid"
    )(channel_attention)
    channel_attention = layers.Reshape(
        (1, 1, channels), name=f"{name}_channel_reshape"
    )(channel_attention)

    x = layers.Multiply(name=f"{name}_channel_multiply")(
        [input_tensor, channel_attention]
    )

    # --- Spatial attention ---
    avg_spatial = ChannelPool(reduction="mean", name=f"{name}_spatial_avg")(x)
    max_spatial = ChannelPool(reduction="max", name=f"{name}_spatial_max")(x)

    spatial = layers.Concatenate(axis=-1, name=f"{name}_spatial_concat")(
        [avg_spatial, max_spatial]
    )
    spatial = layers.Conv2D(
        1, 7, padding="same", activation="sigmoid", name=f"{name}_spatial_conv"
    )(spatial)

    return layers.Multiply(name=f"{name}_spatial_multiply")([x, spatial])
