"""Data generators for the train/validation/test splits.

Two backends are available, selected by ``data.augmentation_backend``:

``keras``
    ``ImageDataGenerator.flow_from_directory``. Simple, but applies the same
    augmentation to every sample.

``albumentations``
    :class:`AlbumentationsGenerator`, which picks the intensive or moderate
    pipeline per sample based on the sample's class. This is what the
    best-performing run uses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dr.config import IMAGE_EXTENSIONS, DataConfig, ExperimentConfig
from dr.data import analysis, augmentation


class Generators(NamedTuple):
    """The three split generators plus the class-index mapping."""

    train: object
    validation: object
    test: object
    class_indices: Dict[str, int]


class AlbumentationsGenerator(tf.keras.utils.Sequence):
    """Directory-backed generator applying a per-class Albumentations pipeline.

    Exposes the same attributes the rest of the pipeline relies on
    (``samples``, ``classes``, ``class_indices``, ``reset``) so it can be swapped
    with a Keras ``DirectoryIterator``.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        default_transform,
        minority_transform=None,
        *,
        img_size: int = 224,
        batch_size: int = 16,
        minority_indices: Iterable[int] = (),
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.default_transform = default_transform
        self.minority_transform = minority_transform or default_transform
        self.img_size = img_size
        self.batch_size = batch_size
        self.minority_indices: Set[int] = set(minority_indices)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.file_paths: List[Path] = []
        labels: List[int] = []
        self.class_indices: Dict[str, int] = {}

        for idx, class_dir in enumerate(
            sorted(p for p in self.directory.iterdir() if p.is_dir())
        ):
            self.class_indices[class_dir.name] = idx
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.file_paths.append(image_path)
                    labels.append(idx)

        if not self.file_paths:
            raise FileNotFoundError(f"No images found under {self.directory}")

        self.classes = np.asarray(labels, dtype=np.int64)
        self.num_classes = len(self.class_indices)
        # Index order is what gets shuffled, so self.classes stays aligned with
        # self.file_paths and evaluation can compare against the true labels.
        self.index = np.arange(len(self.file_paths))
        self.on_epoch_end()

    @property
    def samples(self) -> int:
        """Number of images, matching the Keras generator attribute."""
        return len(self.file_paths)

    def __len__(self) -> int:
        return int(np.ceil(self.samples / self.batch_size))

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        import cv2

        selection = self.index[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        images, labels = [], []

        for i in selection:
            image = cv2.imread(str(self.file_paths[i]))
            if image is None:
                raise OSError(f"Could not read image: {self.file_paths[i]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

            label = int(self.classes[i])
            transform = (
                self.minority_transform
                if label in self.minority_indices
                else self.default_transform
            )
            images.append(transform(image=image)["image"])
            labels.append(label)

        return (
            np.asarray(images, dtype=np.float32),
            tf.keras.utils.to_categorical(labels, num_classes=self.num_classes),
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.index)

    def reset(self) -> None:
        """Restore deterministic ordering before evaluation."""
        self.index = np.arange(len(self.file_paths))

    @property
    def ordered_labels(self) -> np.ndarray:
        """True labels in the current iteration order."""
        return self.classes[self.index]


def create_generators(config: ExperimentConfig) -> Generators:
    """Build the train/validation/test generators described by ``config``."""
    train_counts, minority = analysis.summarise_split(
        config.paths.split("train"), config.data.minority_threshold_ratio
    )

    if config.data.augmentation_backend == "albumentations":
        generators = _albumentations_generators(config, train_counts, minority)
    else:
        generators = _keras_generators(config)

    print("\nGenerators created:")
    print(f"  Train:      {generators.train.samples:,} images")
    print(f"  Validation: {generators.validation.samples:,} images")
    print(f"  Test:       {generators.test.samples:,} images")
    print(f"  Classes:    {generators.class_indices}")
    return generators


def _keras_generators(config: ExperimentConfig) -> Generators:
    data = config.data
    aug = data.keras_augmentation

    if data.use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=aug.rotation_range,
            width_shift_range=aug.width_shift_range,
            height_shift_range=aug.height_shift_range,
            shear_range=aug.shear_range,
            zoom_range=aug.zoom_range,
            brightness_range=aug.brightness_range,
            channel_shift_range=aug.channel_shift_range,
            horizontal_flip=aug.horizontal_flip,
            vertical_flip=aug.vertical_flip,
            fill_mode=aug.fill_mode,
            cval=aug.cval,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    def flow(datagen: ImageDataGenerator, split: str, shuffle: bool):
        return datagen.flow_from_directory(
            config.paths.split(split),
            target_size=(data.img_size, data.img_size),
            batch_size=data.batch_size,
            class_mode="categorical",
            shuffle=shuffle,
            seed=data.seed,
        )

    train = flow(train_datagen, "train", shuffle=True)
    return Generators(
        train=train,
        validation=flow(eval_datagen, "validation", shuffle=False),
        test=flow(eval_datagen, "test", shuffle=False),
        class_indices=train.class_indices,
    )


def _albumentations_generators(
    config: ExperimentConfig,
    train_counts: Dict[str, int],
    minority_classes: Dict[str, int],
) -> Generators:
    data = config.data
    minority_indices = (
        analysis.minority_class_indices(train_counts, data.minority_threshold_ratio)
        if data.use_augmentation and data.use_class_specific_aug
        else set()
    )
    if minority_indices:
        print(
            f"  Intensive augmentation for: {sorted(minority_classes)} "
            f"(indices {sorted(minority_indices)})"
        )

    eval_transform = augmentation.normalisation_only()
    if data.use_augmentation:
        train_default = augmentation.moderate_augmentation()
        train_minority = augmentation.intensive_augmentation()
    else:
        train_default = train_minority = eval_transform

    def build(split: str, default, minority, shuffle: bool, indices: Set[int]):
        return AlbumentationsGenerator(
            config.paths.split(split),
            default,
            minority,
            img_size=data.img_size,
            batch_size=data.batch_size,
            minority_indices=indices,
            shuffle=shuffle,
            seed=data.seed,
        )

    train = build("train", train_default, train_minority, True, minority_indices)
    return Generators(
        train=train,
        validation=build("validation", eval_transform, eval_transform, False, set()),
        test=build("test", eval_transform, eval_transform, False, set()),
        class_indices=train.class_indices,
    )


def compute_class_weights(
    generator, class_names: Sequence[str] | None = None
) -> Tuple[Dict[int, float], np.ndarray]:
    """Balanced class weights derived from a generator's label array."""
    labels = np.asarray(generator.classes)
    weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    weight_dict = {int(i): float(w) for i, w in enumerate(weights)}

    names = list(class_names or sorted(generator.class_indices))
    print("\nClass weights:")
    for name, weight in zip(names, weights):
        print(f"  {name:<20} {weight:.4f}")

    return weight_dict, weights
