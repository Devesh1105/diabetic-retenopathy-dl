#!/usr/bin/env python3
"""Evaluate a saved checkpoint without retraining.

Example:
    python scripts/evaluate.py --config configs/class_specific_aug.yaml \
        --dataset data/processed --split test
"""

from __future__ import annotations

import argparse
import sys

from dr.config import ExperimentConfig
from dr.data.generators import create_generators
from dr.evaluation import (
    evaluate,
    plot_confusion_matrix,
    plot_per_class_predictions,
    plot_roc_curves,
    save_report,
)
from dr.training import load_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument("--dataset", help="Override paths.dataset")
    parser.add_argument("--models-dir", help="Override paths.models")
    parser.add_argument("--results-dir", help="Override paths.results")
    parser.add_argument(
        "--split",
        choices=["validation", "test", "both"],
        default="both",
        help="Which split(s) to evaluate",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "paths.dataset": args.dataset,
            "paths.models": args.models_dir,
            "paths.results": args.results_dir,
        }.items()
        if value is not None
    }
    config = ExperimentConfig.from_yaml(args.config, overrides)

    generators = create_generators(config)
    model = load_checkpoint(config)

    wanted = ["validation", "test"] if args.split == "both" else [args.split]
    results = [
        evaluate(
            model,
            getattr(generators, split),
            config.data.class_names,
            split.capitalize(),
        )
        for split in wanted
    ]

    if not args.no_plots:
        print("\nWriting figures...")
        for result in results:
            plot_confusion_matrix(result, config.paths.results)
            plot_confusion_matrix(result, config.paths.results, normalize=True)
            plot_roc_curves(result, config.paths.results)
            plot_per_class_predictions(result, config.paths.results)

    save_report(config, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
