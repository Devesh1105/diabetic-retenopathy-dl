"""Tests for the dataset inspection helpers. No TensorFlow required."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from dr.config import CLASS_NAMES
from dr.data import analysis


def build_dataset(root: Path, counts: Dict[str, Dict[str, int]]) -> Path:
    """Create a dummy ``split/class/image`` tree with empty .jpg files."""
    for split, class_counts in counts.items():
        for class_name, count in class_counts.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                (class_dir / f"{class_name}_{i}.jpg").touch()
    return root


def test_class_distribution_counts_images(tmp_path: Path) -> None:
    build_dataset(tmp_path, {"train": {"Mild": 3, "No_DR": 7}})

    assert analysis.class_distribution(tmp_path / "train") == {"Mild": 3, "No_DR": 7}


def test_class_distribution_ignores_non_images(tmp_path: Path) -> None:
    build_dataset(tmp_path, {"train": {"Mild": 2}})
    (tmp_path / "train" / "Mild" / "notes.txt").touch()

    assert analysis.class_distribution(tmp_path / "train") == {"Mild": 2}


def test_minority_classes_sit_below_the_threshold() -> None:
    # Average is 100; the 0.7 cutoff is 70, so only Severe qualifies.
    counts = {"No_DR": 200, "Moderate": 100, "Severe": 10, "Mild": 90}

    assert analysis.find_minority_classes(counts, 0.7) == {"Severe": 10}


def test_minority_indices_follow_alphabetical_class_order() -> None:
    counts = {"No_DR": 200, "Moderate": 100, "Severe": 10, "Mild": 90}
    # sorted -> Mild(0), Moderate(1), No_DR(2), Severe(3)

    assert analysis.minority_class_indices(counts, 0.7) == {3}


def test_balanced_dataset_has_no_minority_classes() -> None:
    counts = {name: 100 for name in CLASS_NAMES}

    assert analysis.find_minority_classes(counts) == {}


def test_imbalance_report_scores_a_skewed_dataset(tmp_path: Path) -> None:
    build_dataset(
        tmp_path,
        {
            "train": {"No_DR": 200, "Mild": 20, "Moderate": 60,
                      "Proliferate_DR": 15, "Severe": 10},
            "validation": {"No_DR": 20, "Mild": 2, "Moderate": 6,
                           "Proliferate_DR": 2, "Severe": 1},
            "test": {"No_DR": 30, "Mild": 3, "Moderate": 9,
                     "Proliferate_DR": 3, "Severe": 2},
        },
    )

    report = analysis.imbalance_report(tmp_path)

    assert report["total_images"] == 383
    assert report["imbalance_ratio"] == pytest.approx(250 / 13)
    assert report["severity_level"] in {"high", "critical"}
    assert set(report["per_split"]) == {"train", "validation", "test"}


def test_verify_labels_accepts_a_clean_layout(tmp_path: Path) -> None:
    build_dataset(
        tmp_path,
        {split: {name: 2 for name in CLASS_NAMES}
         for split in ("train", "validation", "test")},
    )

    result = analysis.verify_labels(tmp_path)

    assert result["passed"]
    assert result["labelled_images"] == 30


def test_verify_labels_flags_loose_and_unknown_files(tmp_path: Path) -> None:
    build_dataset(
        tmp_path,
        {split: {name: 2 for name in CLASS_NAMES}
         for split in ("train", "validation", "test")},
    )
    (tmp_path / "train" / "stray.jpg").touch()
    (tmp_path / "test" / "Unknown_Grade").mkdir()
    (tmp_path / "test" / "Unknown_Grade" / "a.jpg").touch()

    result = analysis.verify_labels(tmp_path)

    assert not result["passed"]
    assert len(result["unlabelled_images"]) == 1
    assert result["unknown_classes"] == {"Unknown_Grade": 1}


def test_verify_labels_reports_a_missing_split(tmp_path: Path) -> None:
    build_dataset(tmp_path, {"train": {name: 1 for name in CLASS_NAMES}})

    result = analysis.verify_labels(tmp_path)

    assert any("Missing split folder: validation/" in i for i in result["issues"])
