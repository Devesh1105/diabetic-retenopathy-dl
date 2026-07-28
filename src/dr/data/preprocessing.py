"""Turn the raw Roboflow export into a resized, stratified train/val/test set.

The raw download already ships with splits, but they are small and unevenly
distributed. :func:`prepare_dataset` pools every image, re-splits it with
stratification, resizes to a fixed size and writes the standard
``split/class/image`` layout the training code expects.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split

from dr.config import CLASS_NAMES, IMAGE_EXTENSIONS

# Roboflow exports name the validation split "valid"; accept both.
SOURCE_SPLIT_NAMES = ("train", "test", "validation", "valid")


@dataclass
class ImageRecord:
    """One source image and the label implied by its folder."""

    path: Path
    class_name: str
    original_split: str


def collect_images(source_path: str | os.PathLike[str]) -> List[ImageRecord]:
    """Gather every labelled image across the source dataset's splits."""
    source_path = Path(source_path)
    records: List[ImageRecord] = []

    for split in SOURCE_SPLIT_NAMES:
        split_path = source_path / split
        if not split_path.is_dir():
            continue
        for class_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(
                        ImageRecord(image_path, class_dir.name, split)
                    )

    return records


def stratified_split(
    records: List[ImageRecord],
    test_size: float = 0.15,
    validation_size: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[ImageRecord]]:
    """Split records into train/validation/test, preserving class proportions."""
    if not records:
        raise ValueError("No images to split")
    if test_size + validation_size >= 1.0:
        raise ValueError("test_size + validation_size must be below 1.0")

    labels = [r.class_name for r in records]
    train_val, test = train_test_split(
        records, test_size=test_size, stratify=labels, random_state=seed
    )

    # Re-scale the validation fraction so it stays correct relative to the
    # whole dataset after the test split has been carved out.
    adjusted_val_size = validation_size / (1 - test_size)
    train_val_labels = [r.class_name for r in train_val]
    train, validation = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        stratify=train_val_labels,
        random_state=seed,
    )

    return {"train": train, "validation": validation, "test": test}


def resize_image(
    image_path: str | os.PathLike[str],
    target_size: Tuple[int, int] = (224, 224),
    maintain_aspect: bool = False,
) -> Image.Image:
    """Resize an image, optionally letterboxing it to keep the aspect ratio."""
    image = Image.open(image_path).convert("RGB")

    if not maintain_aspect:
        return image.resize(target_size, Image.LANCZOS)

    image.thumbnail(target_size, Image.LANCZOS)
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    offset = (
        (target_size[0] - image.size[0]) // 2,
        (target_size[1] - image.size[1]) // 2,
    )
    canvas.paste(image, offset)
    return canvas


def write_split(
    records: List[ImageRecord],
    split_name: str,
    output_path: str | os.PathLike[str],
    target_size: Tuple[int, int] = (224, 224),
    maintain_aspect: bool = False,
    quality: int = 95,
) -> Tuple[int, List[str]]:
    """Resize and copy one split. Returns ``(written, failures)``."""
    output_path = Path(output_path)
    written = 0
    failures: List[str] = []

    for record in records:
        destination_dir = output_path / split_name / record.class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        try:
            image = resize_image(record.path, target_size, maintain_aspect)
            image.save(destination_dir / record.path.name, quality=quality)
            written += 1
        except (OSError, ValueError) as exc:
            failures.append(f"{record.path}: {exc}")

    return written, failures


def verify_dimensions(
    dataset_path: str | os.PathLike[str],
    expected_size: Tuple[int, int] = (224, 224),
    sample_size: int = 100,
    seed: int = 42,
) -> Tuple[bool, Dict[Tuple[int, int], int]]:
    """Sample images per class and confirm they all have ``expected_size``."""
    import random

    rng = random.Random(seed)
    dataset_path = Path(dataset_path)
    found: Dict[Tuple[int, int], int] = defaultdict(int)

    for split_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            images = [
                p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            for image_path in rng.sample(images, min(sample_size, len(images))):
                with Image.open(image_path) as image:
                    found[image.size] += 1

    return set(found) <= {expected_size}, dict(found)


def prepare_dataset(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    target_size: Tuple[int, int] = (224, 224),
    test_size: float = 0.15,
    validation_size: float = 0.10,
    maintain_aspect: bool = False,
    seed: int = 42,
) -> Dict[str, object]:
    """Run the full preparation: collect, split, resize and write.

    Returns a summary dict with per-split counts and any failures.
    """
    source_path, output_path = Path(source_path), Path(output_path)
    records = collect_images(source_path)
    if not records:
        raise FileNotFoundError(
            f"No images found under {source_path}. Expected "
            f"{'/'.join(SOURCE_SPLIT_NAMES[:2])}/<class>/<image> sub-folders."
        )

    print(f"Found {len(records):,} images in {source_path}")
    for class_name, count in sorted(Counter(r.class_name for r in records).items()):
        print(f"  {class_name:<20} {count:>6,}")

    unexpected = {r.class_name for r in records} - set(CLASS_NAMES)
    if unexpected:
        print(f"  Warning: unexpected class folders present: {sorted(unexpected)}")

    splits = stratified_split(records, test_size, validation_size, seed)

    summary: Dict[str, object] = {"source": str(source_path), "output": str(output_path)}
    counts: Dict[str, int] = {}
    all_failures: List[str] = []

    for split_name, split_records in splits.items():
        written, failures = write_split(
            split_records, split_name, output_path, target_size, maintain_aspect
        )
        counts[split_name] = written
        all_failures.extend(failures)
        share = len(split_records) / len(records) * 100
        print(f"  {split_name:<11} {written:>6,} images ({share:4.1f}%)")

    if all_failures:
        print(f"\n{len(all_failures)} image(s) failed to process:")
        for failure in all_failures[:10]:
            print(f"  - {failure}")

    summary["counts"] = counts
    summary["failures"] = all_failures
    return summary
