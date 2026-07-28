"""Configuration objects for the diabetic retinopathy pipeline.

Every path and hyper-parameter that used to be hard-coded at the top of a
notebook lives here instead, so an experiment is fully described by a YAML
file under ``configs/``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CLASS_NAMES: List[str] = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
"""Class folder names, in the alphabetical order used by the data generators."""

SPLITS: List[str] = ["train", "validation", "test"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# ImageNet statistics, used by the Albumentations pipelines.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PathConfig:
    """Filesystem locations for a run.

    ``dataset`` is the only required value. It points at a directory holding
    ``train/``, ``validation/`` and ``test/`` sub-directories, each containing
    one folder per class.
    """

    dataset: Path
    models: Path = Path("outputs/models")
    results: Path = Path("outputs/results")

    def __post_init__(self) -> None:
        self.dataset = Path(self.dataset).expanduser()
        self.models = Path(self.models).expanduser()
        self.results = Path(self.results).expanduser()

    def split(self, name: str) -> Path:
        """Return the directory for a split (``train``/``validation``/``test``)."""
        return self.dataset / name

    def create(self) -> None:
        """Create the output directories if they do not exist yet."""
        self.models.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)


@dataclass
class KerasAugmentationConfig:
    """Parameters passed straight to ``ImageDataGenerator``."""

    rotation_range: int = 15
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    shear_range: float = 0.0
    zoom_range: float = 0.1
    brightness_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    channel_shift_range: float = 0.0
    horizontal_flip: bool = True
    vertical_flip: bool = False
    fill_mode: str = "constant"
    cval: float = 0.0


@dataclass
class DataConfig:
    """How images are loaded and augmented."""

    img_size: int = 224
    batch_size: int = 16
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    seed: int = 42

    use_augmentation: bool = True
    # "keras" uses ImageDataGenerator; "albumentations" enables the
    # class-specific pipelines in dr.data.augmentation.
    augmentation_backend: str = "keras"
    # Only meaningful for the albumentations backend: minority classes get the
    # intensive pipeline, everything else the moderate one.
    use_class_specific_aug: bool = True
    # A class is "minority" when it holds fewer than this fraction of the
    # average per-class sample count.
    minority_threshold_ratio: float = 0.7

    keras_augmentation: KerasAugmentationConfig = field(
        default_factory=KerasAugmentationConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.keras_augmentation, dict):
            self.keras_augmentation = KerasAugmentationConfig(**self.keras_augmentation)
        if self.augmentation_backend not in {"keras", "albumentations"}:
            raise ValueError(
                "augmentation_backend must be 'keras' or 'albumentations', "
                f"got {self.augmentation_backend!r}"
            )


@dataclass
class ModelConfig:
    """Attention U-Net architecture options."""

    encoder_name: str = "resnet50"  # "resnet50" or "efficientnetb0"
    pretrained_encoder: bool = True
    dropout_rate: float = 0.5
    dense_units: int = 512
    cbam_reduction: int = 16


@dataclass
class TrainingConfig:
    """Optimisation, loss and callback settings."""

    num_epochs: int = 30
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7

    use_class_weights: bool = True
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    checkpoint_filename: str = "best_model.keras"


@dataclass
class ExperimentConfig:
    """Top-level config: everything needed to reproduce one training run."""

    name: str
    paths: PathConfig
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    description: str = ""

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict) and f.name in _SECTION_TYPES:
                setattr(self, f.name, _SECTION_TYPES[f.name](**value))

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike[str], overrides: Optional[Dict[str, Any]] = None
    ) -> "ExperimentConfig":
        """Load a config from YAML, optionally applying dotted-key overrides.

        Overrides use the same dotted notation as the CLI flags, e.g.
        ``{"training.num_epochs": 5, "paths.dataset": "/data/dr"}``.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        raw.setdefault("name", Path(path).stem)
        for key, value in (overrides or {}).items():
            _set_dotted(raw, key, value)

        env_dataset = os.environ.get("DR_DATASET_PATH")
        if env_dataset and not raw.get("paths", {}).get("dataset"):
            raw.setdefault("paths", {})["dataset"] = env_dataset

        if not raw.get("paths", {}).get("dataset"):
            raise ValueError(
                "No dataset path configured. Set paths.dataset in the YAML file, "
                "pass --dataset, or export DR_DATASET_PATH."
            )
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain Python types (paths become strings)."""
        return _stringify(asdict(self))

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the resolved config next to a run's outputs."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


_SECTION_TYPES = {
    "paths": PathConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
}


def _set_dotted(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def _stringify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
