"""Attention U-Net with a pretrained encoder, adapted for classification.

The decoder is the standard attention U-Net path, but instead of a
segmentation head the final feature map goes through CBAM, global average
pooling and a dense classifier that outputs one probability per DR grade.
"""

from __future__ import annotations

from typing import List, Tuple

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50

from dr.config import DataConfig, ModelConfig
from dr.models.attention import attention_gate, cbam_block

# Encoder layers tapped for skip connections, coarsest last.
RESNET50_SKIP_LAYERS = (
    "conv1_relu",  # 112x112x64
    "conv2_block3_out",  # 56x56x256
    "conv3_block4_out",  # 28x28x512
    "conv4_block6_out",  # 14x14x1024
    "conv5_block3_out",  # 7x7x2048
)

EFFICIENTNETB0_SKIP_LAYERS = (
    "block2a_expand_activation",
    "block3a_expand_activation",
    "block4a_expand_activation",
    "block6a_expand_activation",
)


def _build_encoder(inputs, model_config: ModelConfig) -> Tuple[List, List[int]]:
    """Return the encoder skip tensors (fine -> coarse) and their widths."""
    encoder = model_config.encoder_name.lower()
    weights = "imagenet" if model_config.pretrained_encoder else None

    if encoder == "resnet50":
        base = ResNet50(weights=weights, include_top=False, input_tensor=inputs)
        skips = [base.get_layer(name).output for name in RESNET50_SKIP_LAYERS]
    elif encoder == "efficientnetb0":
        base = EfficientNetB0(weights=weights, include_top=False, input_tensor=inputs)
        skips = [
            base.get_layer(name).output for name in EFFICIENTNETB0_SKIP_LAYERS
        ] + [base.output]
    else:
        raise ValueError(
            f"Unsupported encoder {model_config.encoder_name!r}. "
            "Choose 'resnet50' or 'efficientnetb0'."
        )

    return skips, [int(skip.shape[-1]) for skip in skips]


def _decoder_block(x, skip, filters: int, level: int):
    """Upsample, gate the skip connection, concatenate and refine."""
    x = layers.UpSampling2D(size=(2, 2), name=f"upsample{level}")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"d{level}_conv1")(x)
    x = layers.BatchNormalization(name=f"d{level}_bn1")(x)
    x = layers.Activation("relu", name=f"d{level}_relu1")(x)

    gated_skip = attention_gate(x, skip, max(filters // 2, 1), name=f"att{level}")
    x = layers.Concatenate(name=f"concat{level}")([x, gated_skip])

    x = layers.Conv2D(filters, 3, padding="same", name=f"d{level}_conv2")(x)
    x = layers.BatchNormalization(name=f"d{level}_bn2")(x)
    return layers.Activation("relu", name=f"d{level}_relu2")(x)


def build_attention_unet(
    model_config: ModelConfig, data_config: DataConfig
) -> Model:
    """Build the Attention U-Net classifier.

    Args:
        model_config: Encoder, dropout and head settings.
        data_config: Supplies the input resolution and number of classes.

    Returns:
        An uncompiled ``tf.keras.Model`` producing softmax class scores.
    """
    inputs = layers.Input(
        shape=(data_config.img_size, data_config.img_size, 3), name="image"
    )
    skips, channels = _build_encoder(inputs, model_config)
    e1, e2, e3, e4, bottleneck = skips

    # Decoder levels are numbered to match the encoder stage they consume, so
    # layer names line up with the feature-map visualisations.
    x = _decoder_block(bottleneck, e4, channels[3], level=5)
    x = _decoder_block(x, e3, channels[2], level=4)
    x = _decoder_block(x, e2, channels[1], level=3)
    x = _decoder_block(x, e1, channels[0], level=2)

    features = cbam_block(x, reduction=model_config.cbam_reduction, name="cbam")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.Dropout(model_config.dropout_rate, name="dropout1")(x)
    x = layers.Dense(model_config.dense_units, activation="relu", name="fc1")(x)
    x = layers.Dropout(model_config.dropout_rate * 0.6, name="dropout2")(x)
    outputs = layers.Dense(
        data_config.num_classes, activation="softmax", name="predictions"
    )(x)

    return Model(inputs=inputs, outputs=outputs, name="AttentionUNet")
