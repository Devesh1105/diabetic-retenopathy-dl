"""Tests for dataset preparation.

Skipped automatically when scikit-learn/Pillow are not installed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from dr.config import CLASS_NAMES  # noqa: E402
from dr.data import preprocessing  # noqa: E402


def build_raw_dataset(root: Path, per_class: int = 20) -> Path:
    """A Roboflow-style export with a 'valid' split and varied image sizes."""
    for split, share in (("train", 0.7), ("valid", 0.15), ("test", 0.15)):
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(max(int(per_class * share), 1)):
                size = (320 + class_index * 10, 240 + i)
                Image.new("RGB", size, (i % 255, 0, 0)).save(
                    class_dir / f"{class_name}_{split}_{i}.jpg"
                )
    return root


def test_collect_images_reads_every_split(tmp_path: Path) -> None:
    build_raw_dataset(tmp_path)

    records = preprocessing.collect_images(tmp_path)

    assert len(records) > 0
    assert {r.class_name for r in records} == set(CLASS_NAMES)
    # "valid" is the Roboflow name for the validation split and must be picked up.
    assert "valid" in {r.original_split for r in records}


def test_stratified_split_preserves_class_proportions(tmp_path: Path) -> None:
    build_raw_dataset(tmp_path, per_class=40)
    records = preprocessing.collect_images(tmp_path)

    splits = preprocessing.stratified_split(records, test_size=0.15, validation_size=0.1)

    assert set(splits) == {"train", "validation", "test"}
    assert sum(len(v) for v in splits.values()) == len(records)
    for split_records in splits.values():
        assert set(Counter(r.class_name for r in split_records)) == set(CLASS_NAMES)


def test_stratified_split_rejects_impossible_fractions(tmp_path: Path) -> None:
    build_raw_dataset(tmp_path)
    records = preprocessing.collect_images(tmp_path)

    with pytest.raises(ValueError):
        preprocessing.stratified_split(records, test_size=0.8, validation_size=0.3)


def test_resize_image_stretches_to_target(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    Image.new("RGB", (640, 200), (10, 20, 30)).save(source)

    assert preprocessing.resize_image(source, (224, 224)).size == (224, 224)


def test_resize_image_letterboxes_when_keeping_aspect(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    Image.new("RGB", (640, 200), (255, 255, 255)).save(source)

    resized = preprocessing.resize_image(source, (224, 224), maintain_aspect=True)

    assert resized.size == (224, 224)
    # A wide image padded into a square leaves black bars at the top.
    assert resized.getpixel((112, 2)) == (0, 0, 0)


def test_prepare_dataset_writes_a_verifiable_tree(tmp_path: Path) -> None:
    source = build_raw_dataset(tmp_path / "raw", per_class=40)
    output = tmp_path / "processed"

    summary = preprocessing.prepare_dataset(source, output, target_size=(64, 64))

    assert not summary["failures"]
    assert sum(summary["counts"].values()) == len(
        preprocessing.collect_images(source)
    )

    ok, found = preprocessing.verify_dimensions(output, (64, 64))
    assert ok, f"unexpected sizes: {found}"


def test_prepare_dataset_rejects_an_empty_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preprocessing.prepare_dataset(tmp_path, tmp_path / "out")
