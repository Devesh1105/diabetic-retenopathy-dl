"""Diabetic retinopathy severity classification with an Attention U-Net.

The package mirrors the stages of the pipeline:

``dr.config``      configuration dataclasses loaded from ``configs/*.yaml``
``dr.data``        dataset preparation, augmentation and generators
``dr.models``      attention blocks and the Attention U-Net classifier
``dr.training``    losses, metrics, callbacks and the training loop
``dr.evaluation``  metrics, plots and run reports
"""

from dr.config import (
    CLASS_NAMES,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
)

__all__ = [
    "CLASS_NAMES",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PathConfig",
    "TrainingConfig",
]

__version__ = "0.1.0"
