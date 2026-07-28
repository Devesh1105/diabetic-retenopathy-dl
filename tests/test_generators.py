"""Tests for the data generators against a small synthetic dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("tensorflow")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dr.config import DataConfig, ExperimentConfig, PathConfig  # noqa: E402
from dr.data.generators import compute_class_weights, create_generators  # noqa: E402

IMG_SIZE = 32
# Deliberately lopsided: Severe/Proliferate_DR fall below the minority cutoff.
TRAIN_COUNTS = {
    "Mild": 8,
    "Moderate": 12,
    "No_DR": 24,
    "Proliferate_DR": 4,
    "Severe": 2,
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(0)
    for split, scale in (("train", 1.0), ("validation", 0.5), ("test", 0.5)):
        for class_name, count in TRAIN_COUNTS.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(max(int(count * scale), 1)):
                pixels = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"{class_name}_{i}.png")
    return root


def make_config(dataset: Path, **data_kwargs) -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        paths=PathConfig(dataset=dataset),
        data=DataConfig(img_size=IMG_SIZE, batch_size=4, **data_kwargs),
    )


def test_keras_generators_yield_correctly_shaped_batches(dataset: Path) -> None:
    generators = create_generators(make_config(dataset, augmentation_backend="keras"))

    images, labels = generators.train[0]

    assert images.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
    assert labels.shape[1] == 5
    assert generators.train.samples == sum(TRAIN_COUNTS.values())
    assert list(generators.class_indices) == sorted(TRAIN_COUNTS)


def test_albumentations_generators_yield_correctly_shaped_batches(
    dataset: Path,
) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    generators = create_generators(
        make_config(dataset, augmentation_backend="albumentations")
    )

    images, labels = generators.train[0]

    assert images.shape == (4, IMG_SIZE, IMG_SIZE, 3)
    assert labels.shape == (4, 5)
    assert generators.train.samples == sum(TRAIN_COUNTS.values())


def test_minority_classes_get_the_intensive_pipeline(dataset: Path) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    generators = create_generators(
        make_config(dataset, augmentation_backend="albumentations")
    )

    # Average count is 10, so the 0.7 cutoff is 7: Severe (2) and
    # Proliferate_DR (4) qualify. Sorted order puts them at indices 3 and 4.
    assert generators.train.minority_indices == {3, 4}
    # Evaluation splits are never augmented.
    assert generators.test.minority_indices == set()


def test_evaluation_generators_keep_labels_aligned_after_reset(dataset: Path) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    generators = create_generators(
        make_config(dataset, augmentation_backend="albumentations")
    )
    test_gen = generators.test

    batch_labels = np.argmax(test_gen[0][1], axis=1)

    assert np.array_equal(batch_labels, test_gen.ordered_labels[: len(batch_labels)])


def test_shuffling_reorders_samples_without_desyncing_labels(dataset: Path) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")
    generators = create_generators(
        make_config(dataset, augmentation_backend="albumentations")
    )
    train_gen = generators.train

    train_gen.on_epoch_end()
    shuffled = train_gen.ordered_labels.copy()
    train_gen.reset()

    assert sorted(shuffled) == sorted(train_gen.ordered_labels)
    assert np.array_equal(train_gen.ordered_labels, train_gen.classes)


def test_class_weights_favour_the_rare_grades(dataset: Path) -> None:
    generators = create_generators(make_config(dataset, augmentation_backend="keras"))

    weight_dict, weights = compute_class_weights(
        generators.train, sorted(TRAIN_COUNTS)
    )

    assert len(weight_dict) == 5
    # Severe is the rarest class and must carry the largest weight.
    assert weights.argmax() == sorted(TRAIN_COUNTS).index("Severe")
    assert weights.argmin() == sorted(TRAIN_COUNTS).index("No_DR")


def test_unknown_backend_is_rejected(dataset: Path) -> None:
    with pytest.raises(ValueError, match="augmentation_backend"):
        make_config(dataset, augmentation_backend="torchvision")


def test_missing_split_raises_a_clear_error(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()

    with pytest.raises(FileNotFoundError):
        create_generators(make_config(tmp_path, augmentation_backend="keras"))
