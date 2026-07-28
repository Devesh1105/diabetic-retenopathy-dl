#!/usr/bin/env python3
"""Train one experiment and evaluate it on validation and test.

Example:
    python scripts/train.py --config configs/class_specific_aug.yaml \
        --dataset data/processed
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from dr.config import ExperimentConfig
from dr.evaluation import (
    evaluate,
    plot_confusion_matrix,
    plot_per_class_predictions,
    plot_roc_curves,
    plot_sample_predictions,
    plot_training_history,
    save_report,
)
from dr.training import train


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument("--dataset", help="Override paths.dataset")
    parser.add_argument("--models-dir", help="Override paths.models")
    parser.add_argument("--results-dir", help="Override paths.results")
    parser.add_argument("--epochs", type=int, help="Override training.num_epochs")
    parser.add_argument("--batch-size", type=int, help="Override data.batch_size")
    parser.add_argument(
        "--learning-rate", type=float, help="Override training.learning_rate"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figure generation"
    )
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "paths.dataset": args.dataset,
        "paths.models": args.models_dir,
        "paths.results": args.results_dir,
        "training.num_epochs": args.epochs,
        "training.learning_rate": args.learning_rate,
        "data.batch_size": args.batch_size,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config, build_overrides(args))

    model, history, generators, class_weights = train(config)

    results = [
        evaluate(model, generators.validation, config.data.class_names, "Validation"),
        evaluate(model, generators.test, config.data.class_names, "Test"),
    ]

    if not args.no_plots:
        print("\nWriting figures...")
        plot_training_history(history, config.paths.results)
        for result in results:
            plot_confusion_matrix(result, config.paths.results)
            plot_confusion_matrix(result, config.paths.results, normalize=True)
            plot_roc_curves(result, config.paths.results)
        plot_per_class_predictions(results[-1], config.paths.results)
        plot_sample_predictions(
            model, generators.test, config.data.class_names, config.paths.results
        )

    save_report(config, results, history, class_weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
