"""Dataset inspection: class counts, imbalance metrics and label checks."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dr.config import CLASS_NAMES, IMAGE_EXTENSIONS, SPLITS


def count_images(directory: str | os.PathLike[str]) -> int:
    """Count image files directly inside ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(
        1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )


def class_distribution(split_path: str | os.PathLike[str]) -> Dict[str, int]:
    """Map class folder name -> number of images, for one split."""
    split_path = Path(split_path)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_path}")
    return {
        entry.name: count_images(entry)
        for entry in sorted(split_path.iterdir())
        if entry.is_dir()
    }


def find_minority_classes(
    class_counts: Dict[str, int], threshold_ratio: float = 0.7
) -> Dict[str, int]:
    """Classes holding fewer than ``threshold_ratio`` x the per-class average.

    These get the intensive augmentation pipeline during training.
    """
    if not class_counts:
        return {}
    average = sum(class_counts.values()) / len(class_counts)
    threshold = average * threshold_ratio
    return {name: n for name, n in class_counts.items() if n < threshold}


def minority_class_indices(
    class_counts: Dict[str, int], threshold_ratio: float = 0.7
) -> Set[int]:
    """Minority classes as generator label indices (alphabetical order)."""
    minority = find_minority_classes(class_counts, threshold_ratio)
    ordered = sorted(class_counts)
    return {idx for idx, name in enumerate(ordered) if name in minority}


def summarise_split(
    split_path: str | os.PathLike[str], threshold_ratio: float = 0.7
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Print and return ``(class_counts, minority_classes)`` for a split."""
    class_counts = class_distribution(split_path)
    minority = find_minority_classes(class_counts, threshold_ratio)

    total = sum(class_counts.values())
    average = total / len(class_counts) if class_counts else 0

    print("\nClass distribution:")
    print(f"  Total samples:     {total:,}")
    print(f"  Average per class: {average:,.1f}")
    print(f"  Minority cutoff:   {average * threshold_ratio:,.1f}")
    for name, count in sorted(class_counts.items(), key=lambda kv: kv[1]):
        tag = "MINORITY" if name in minority else ""
        print(f"    {name:<20} {count:>6,} {tag}")

    return class_counts, minority


def imbalance_report(dataset_path: str | os.PathLike[str]) -> Dict[str, object]:
    """Aggregate class counts across splits and score the imbalance.

    The severity score (0-100) combines the majority/minority ratio, the
    absolute size of the smallest class, and how many classes sit below half
    the balanced share.
    """
    dataset_path = Path(dataset_path)
    per_split: Dict[str, Dict[str, int]] = {}
    totals: Dict[str, int] = defaultdict(int)

    for split in SPLITS:
        split_path = dataset_path / split
        if not split_path.is_dir():
            continue
        counts = class_distribution(split_path)
        per_split[split] = counts
        for name, count in counts.items():
            totals[name] += count

    if not totals:
        raise FileNotFoundError(f"No class folders found under {dataset_path}")

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    max_count, min_count = ordered[0][1], ordered[-1][1]
    total_images = sum(totals.values())
    ratio = max_count / min_count if min_count else float("inf")

    score = 0
    for bound, points in ((10, 40), (5, 30), (3, 20), (2, 10)):
        if ratio >= bound:
            score += points
            break
    for bound, points in ((500, 30), (1000, 20), (2000, 10)):
        if min_count < bound:
            score += points
            break
    balanced_share = total_images / len(ordered)
    underrepresented = sum(1 for _, c in ordered if c < balanced_share * 0.5)
    score += min(underrepresented * 10, 30)

    if score >= 70:
        severity = "critical"
    elif score >= 50:
        severity = "high"
    elif score >= 30:
        severity = "moderate"
    else:
        severity = "low"

    return {
        "total_images": total_images,
        "class_counts": dict(ordered),
        "per_split": per_split,
        "imbalance_ratio": ratio,
        "severity_score": score,
        "severity_level": severity,
        "inverse_frequency_weights": {
            name: total_images / (len(ordered) * count) for name, count in ordered
        },
    }


def verify_labels(
    dataset_path: str | os.PathLike[str],
    expected_classes: List[str] | None = None,
) -> Dict[str, object]:
    """Check that every image sits in a recognised ``split/class`` folder.

    Returns a dict with the counts plus a list of human-readable issues; an
    empty ``issues`` list means the layout is clean.
    """
    dataset_path = Path(dataset_path)
    expected = set(expected_classes or CLASS_NAMES)

    labelled = 0
    unlabelled: List[str] = []
    unknown_classes: Dict[str, int] = defaultdict(int)
    stats: Dict[str, Dict[str, int]] = {}
    issues: List[str] = []

    for split in SPLITS:
        split_path = dataset_path / split
        if not split_path.is_dir():
            issues.append(f"Missing split folder: {split}/")
            continue

        loose = [
            f for f in split_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if loose:
            unlabelled.extend(str(f) for f in loose)
            issues.append(
                f"{len(loose)} image(s) sit directly in {split}/ "
                "instead of a class folder"
            )

        split_stats: Dict[str, int] = {}
        for folder in sorted(p for p in split_path.iterdir() if p.is_dir()):
            count = count_images(folder)
            split_stats[folder.name] = count
            if folder.name not in expected:
                unknown_classes[folder.name] += count
                issues.append(
                    f"Unexpected class folder {split}/{folder.name} ({count} images)"
                )
            labelled += count

        for class_name in sorted(expected):
            if split_stats.get(class_name, 0) == 0:
                issues.append(f"No images for {split}/{class_name}")

        stats[split] = split_stats

    return {
        "labelled_images": labelled,
        "unlabelled_images": unlabelled,
        "unknown_classes": dict(unknown_classes),
        "stats": stats,
        "issues": issues,
        "passed": not issues,
    }
