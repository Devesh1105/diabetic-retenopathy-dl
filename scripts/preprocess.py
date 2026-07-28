#!/usr/bin/env python3
"""Prepare the raw dataset: re-split, resize and verify.

Example:
    python scripts/preprocess.py \
        --source data/raw/diabetic-retinopathy-v8i \
        --output data/processed
"""

from __future__ import annotations

import argparse
import sys

from dr.data import analysis, preprocessing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Raw dataset root (contains train/test/valid)"
    )
    parser.add_argument("--output", required=True, help="Where to write the splits")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument(
        "--maintain-aspect",
        action="store_true",
        help="Letterbox instead of stretching to the target size",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_size = (args.img_size, args.img_size)

    preprocessing.prepare_dataset(
        source_path=args.source,
        output_path=args.output,
        target_size=target_size,
        test_size=args.test_size,
        validation_size=args.validation_size,
        maintain_aspect=args.maintain_aspect,
        seed=args.seed,
    )

    print("\nVerifying labels...")
    verification = analysis.verify_labels(args.output)
    for issue in verification["issues"]:
        print(f"  ! {issue}")
    print("  labels OK" if verification["passed"] else "  label check FAILED")

    print("\nVerifying dimensions...")
    ok, found = preprocessing.verify_dimensions(args.output, target_size)
    for size, count in sorted(found.items(), key=lambda kv: -kv[1]):
        print(f"  {size}: {count} sampled image(s)")
    print(f"  all images are {target_size}" if ok else "  dimension check FAILED")

    report = analysis.imbalance_report(args.output)
    print(
        f"\nImbalance ratio {report['imbalance_ratio']:.2f}:1 "
        f"(severity: {report['severity_level']}, "
        f"score {report['severity_score']}/100)"
    )

    return 0 if verification["passed"] and ok else 1


if __name__ == "__main__":
    sys.exit(main())
