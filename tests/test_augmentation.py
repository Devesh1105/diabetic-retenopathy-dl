"""Tests for the Albumentations pipelines."""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("albumentations")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from dr.data import augmentation  # noqa: E402

PIPELINES = ["normalisation_only", "moderate_augmentation", "intensive_augmentation"]


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8)


@pytest.mark.parametrize("pipeline_name", PIPELINES)
def test_pipeline_preserves_shape(pipeline_name: str, image: np.ndarray) -> None:
    pipeline = getattr(augmentation, pipeline_name)()

    output = pipeline(image=image)["image"]

    assert output.shape == image.shape
    assert output.dtype == np.float32


@pytest.mark.parametrize("pipeline_name", PIPELINES)
def test_pipeline_builds_without_deprecation_warnings(pipeline_name: str) -> None:
    """Guards against transform arguments being silently ignored on upgrade."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(augmentation, pipeline_name)()

    offending = [
        str(w.message)
        for w in caught
        if "deprecated" in str(w.message).lower()
        or "is not valid and will be ignored" in str(w.message)
    ]
    assert not offending, offending


def test_normalisation_is_deterministic(image: np.ndarray) -> None:
    pipeline = augmentation.normalisation_only()

    first = pipeline(image=image)["image"]
    second = pipeline(image=image)["image"]

    assert np.array_equal(first, second)


def test_augmentation_actually_changes_the_image(image: np.ndarray) -> None:
    baseline = augmentation.normalisation_only()(image=image)["image"]
    augmented = augmentation.intensive_augmentation()(image=image)["image"]

    assert not np.array_equal(baseline, augmented)


def test_std_range_converts_variance_to_a_fraction_of_255() -> None:
    low, high = augmentation._std_range((100.0, 400.0))

    # sqrt(100)/255 and sqrt(400)/255
    assert low == pytest.approx(10 / 255)
    assert high == pytest.approx(20 / 255)
