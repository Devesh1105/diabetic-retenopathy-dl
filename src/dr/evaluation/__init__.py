"""Evaluation metrics, figures and run reports."""

from dr.evaluation.evaluate import EvaluationResult, evaluate, per_class_accuracy
from dr.evaluation.report import build_report, save_report
from dr.evaluation.visualize import (
    plot_confusion_matrix,
    plot_feature_maps,
    plot_per_class_predictions,
    plot_roc_curves,
    plot_sample_predictions,
    plot_training_history,
)

__all__ = [
    "EvaluationResult",
    "build_report",
    "evaluate",
    "per_class_accuracy",
    "plot_confusion_matrix",
    "plot_feature_maps",
    "plot_per_class_predictions",
    "plot_roc_curves",
    "plot_sample_predictions",
    "plot_training_history",
    "save_report",
]
