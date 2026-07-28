"""Run reports: a human-readable summary and a machine-readable JSON dump."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from dr.config import ExperimentConfig
from dr.evaluation.evaluate import EvaluationResult, per_class_accuracy


def build_report(
    config: ExperimentConfig,
    results: Iterable[EvaluationResult],
    history: Optional[object] = None,
    class_weights: Optional[Dict[int, float]] = None,
) -> str:
    """Render the text report for one run."""
    lines = [
        "=" * 70,
        f"ATTENTION U-NET - RUN REPORT: {config.name}",
        "=" * 70,
    ]
    if config.description:
        lines += ["", config.description]

    lines += [
        "",
        "CONFIGURATION",
        f"  Encoder:          {config.model.encoder_name} "
        f"(pretrained={config.model.pretrained_encoder})",
        f"  Image size:       {config.data.img_size}x{config.data.img_size}",
        f"  Batch size:       {config.data.batch_size}",
        f"  Epochs (max):     {config.training.num_epochs}",
        f"  Learning rate:    {config.training.learning_rate}",
        f"  Dropout:          {config.model.dropout_rate}",
        f"  Loss:             "
        f"{'focal' if config.training.use_focal_loss else 'categorical cross-entropy'}",
        f"  Class weights:    {'on' if config.training.use_class_weights else 'off'}",
        f"  Augmentation:     {config.data.augmentation_backend}"
        f"{' (class-specific)' if config.data.use_class_specific_aug else ''}",
    ]

    if class_weights:
        lines += ["", "CLASS WEIGHTS"]
        for index, name in enumerate(config.data.class_names):
            if index in class_weights:
                lines.append(f"  {name:<20} {class_weights[index]:.4f}")

    records = getattr(history, "history", history) or {}
    if records:
        lines += [
            "",
            "TRAINING",
            f"  Epochs run:            {len(records.get('loss', []))}",
            f"  Best train loss:       {min(records.get('loss', [float('nan')])):.4f}",
            f"  Best val loss:         {min(records.get('val_loss', [float('nan')])):.4f}",
            f"  Best val accuracy:     "
            f"{max(records.get('val_accuracy', [0])) * 100:.2f}%",
        ]
        if "val_balanced_accuracy" in records:
            lines.append(
                "  Best val balanced acc: "
                f"{max(records['val_balanced_accuracy']) * 100:.2f}%"
            )

    for result in results:
        lines += ["", f"{result.split_name.upper()} SET"]
        if result.loss == result.loss:  # not NaN
            lines.append(f"  Loss:               {result.loss:.4f}")
        for key, value in result.metrics.items():
            lines.append(f"  {key.replace('_', ' ').capitalize():<19} {value:.4f}")
        lines.append("  Per-class recall:")
        for class_name, recall in per_class_accuracy(result).items():
            lines.append(f"    {class_name:<20} {recall * 100:6.2f}%")

    lines += [
        "",
        "ARTEFACTS",
        f"  Checkpoint: {config.paths.models / config.training.checkpoint_filename}",
        f"  Results:    {config.paths.results}",
        f"  TensorBoard: tensorboard --logdir {config.paths.results / 'logs'}",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


def save_report(
    config: ExperimentConfig,
    results: Iterable[EvaluationResult],
    history: Optional[object] = None,
    class_weights: Optional[Dict[int, float]] = None,
    output_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Write ``report.txt`` and ``results.json`` for a run."""
    results = list(results)
    output_dir = Path(output_dir or config.paths.results)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_text = build_report(config, results, history, class_weights)
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)

    records = getattr(history, "history", history) or {}
    payload = {
        "name": config.name,
        "config": config.to_dict(),
        "epochs_run": len(records.get("loss", [])),
        "class_weights": {str(k): v for k, v in (class_weights or {}).items()},
        "splits": {
            result.split_name.lower(): {
                "loss": result.loss,
                **result.metrics,
                "per_class_recall": per_class_accuracy(result),
                "confusion_matrix": result.confusion_matrix.tolist(),
            }
            for result in results
        },
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nReport written to {report_path}")
    return report_path
